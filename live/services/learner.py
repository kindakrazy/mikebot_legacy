# mikebot/live/services/learner.py

from __future__ import annotations

import logging
import datetime as dt
from datetime import datetime
from typing import Dict, Any, Optional, List, Sequence
from threading import Lock

import pandas as pd

from mikebot.core.experience_store import ExperienceStore
from mikebot.core.self_audit import SelfAudit
from mikebot.core.meta_trainer import MetaTrainer
from mikebot.models.model_registry_v4 import ModelRegistryV4

logger = logging.getLogger(__name__)


class LearnerService:
    """
    The Live Feedback Loop Service (Symbol-Level MULTITF, multi-engine).

    Responsibilities:
    1. Capture live features and predictions per timeframe.
    2. Match predictions with market outcomes (y) as they arrive.
    3. Persist 'experience' into the ExperienceStore.
    4. Run SelfAudit on the symbol level (MULTITF).
    5. Trigger MetaTrainer if Drift > Threshold AND Samples >= N.
    6. For each user-selected engine (rf, xgb, ...), evolve independently.
    """

    def __init__(
        self,
        experience_store: ExperienceStore,
        self_audit: SelfAudit,
        meta_trainer: MetaTrainer,
        model_registry: ModelRegistryV4,
        drift_threshold: float = 0.15,
        retrain_min_samples: int = 200,
        retrain_min_interval: dt.timedelta = dt.timedelta(hours=1),
        engines: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Parameters
        ----------
        experience_store : ExperienceStore
            Persistent store for features/labels/predictions/errors/regimes.
        self_audit : SelfAudit
            Audit component used to compute drift and performance.
        meta_trainer : MetaTrainer
            Symbol-level MULTITF trainer (v4).
        model_registry : ModelRegistryV4
            Symbol-level registry (v4). Used by MetaTrainer; not directly here.
        drift_threshold : float
            Drift threshold above which evolution is triggered.
        retrain_min_samples : int
            Minimum new samples since last retrain before evolution is considered.
        retrain_min_interval : timedelta
            Minimum wall-clock time between retrains for a symbol.
        engines : Sequence[str], optional
            User-selected engines to evolve (e.g., ["rf", "xgb"]).
            If None or empty, defaults to ["rf"].
        """
        self.store = experience_store
        self.audit = self_audit
        self.trainer = meta_trainer
        self.registry = model_registry

        self.drift_threshold = drift_threshold
        self.retrain_min_samples = retrain_min_samples
        self.retrain_min_interval = retrain_min_interval

        # Engines to evolve per symbol (Option A: evolve all selected engines)
        self._engines: List[str] = list(engines) if engines else ["rf"]

        # Thread safety for concurrent TF updates
        self._lock = Lock()

        # State tracking: strictly symbol-level
        self._last_retrain_at: Dict[str, dt.datetime] = {}
        self._new_samples_since_retrain: Dict[str, int] = {}
        self._is_evolving: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Engine helpers
    # ------------------------------------------------------------------

    @property
    def engines(self) -> List[str]:
        """Return the list of engines to evolve (rf, xgb, ...)."""
        return list(self._engines) if self._engines else ["rf"]

    # ------------------------------------------------------------------
    # LIVE INGESTION
    # ------------------------------------------------------------------

    def record_prediction(
        self,
        symbol: str,
        timeframe: str,
        features: pd.DataFrame,
        prediction: float,
        model_version: str,
        regime_info: Dict[str, Any],
    ) -> None:
        """Called immediately after model inference for a single timeframe."""
        if features is None or features.empty:
            logger.warning(f"Learner: Empty features for {symbol} {timeframe}")
            return

        try:
            timestamp = features.index[-1]
            feat_row = features.tail(1)

            pred_df = pd.DataFrame(
                {"prediction": [prediction], "version": [model_version]},
                index=[timestamp],
            )
            regime_df = pd.DataFrame([regime_info], index=[timestamp])

            # Granular storage (per TF)
            self.store.append(
                symbol=symbol,
                timeframe=timeframe,
                features=feat_row,
                predictions=pred_df,
                regimes=regime_df,
            )

            # Symbol-level counter increment
            with self._lock:
                self._new_samples_since_retrain[symbol] = (
                    self._new_samples_since_retrain.get(symbol, 0) + 1
                )

            logger.debug(
                f"Learner: Recorded prediction for {symbol} {timeframe} (v={model_version})"
            )

        except Exception as e:
            logger.error(f"Learner: Error recording prediction for {symbol}: {e}")

    def record_outcome(
        self,
        symbol: str,
        timeframe: str,
        timestamp: datetime,
        actual_y: float,
        features_by_symbol: Dict[str, pd.DataFrame],
    ) -> None:
        """
        Records outcome for a given (symbol, timeframe, timestamp) and
        evaluates symbol-level evolution logic across all selected engines.
        """
        try:
            label_df = pd.DataFrame({"label": [actual_y]}, index=[timestamp])

            # Retrieve prediction for error calculation
            all_data = self.store.load_all(symbol, timeframe)
            preds = all_data.get("predictions") if all_data is not None else None

            if preds is not None and not preds.empty and timestamp in preds.index:
                prediction = float(preds.loc[timestamp, "prediction"])
                error_df = pd.DataFrame(
                    {"abs_error": [abs(actual_y - prediction)]},
                    index=[timestamp],
                )
                self.store.append(
                    symbol=symbol,
                    timeframe=timeframe,
                    labels=label_df,
                    errors=error_df,
                )
            else:
                self.store.append(
                    symbol=symbol,
                    timeframe=timeframe,
                    labels=label_df,
                )

            # Evaluate evolution for the whole symbol (MULTITF)
            self._evaluate_and_maybe_evolve(symbol, features_by_symbol)

        except Exception as e:
            logger.error(f"Learner: Error recording outcome for {symbol}: {e}")

    # ------------------------------------------------------------------
    # EVOLUTION LOGIC
    # ------------------------------------------------------------------

    def _evaluate_and_maybe_evolve(
        self,
        symbol: str,
        features_by_symbol: Dict[str, pd.DataFrame],
    ) -> None:
        """
        Internal trigger check: Drift + Samples + Interval.
        If triggered, evolve all selected engines for this symbol.
        """
        now = dt.datetime.utcnow()

        # Filter active timeframes from current context
        active_tfs: List[str] = [
            tf for tf, feats in features_by_symbol.items()
            if feats is not None and not feats.empty
        ]
        if not active_tfs:
            logger.warning(f"Learner: No active timeframes for {symbol}; skipping evolution check")
            return

        with self._lock:
            samples = self._new_samples_since_retrain.get(symbol, 0)
            last_retrain = self._last_retrain_at.get(symbol)
            is_evolving = self._is_evolving.get(symbol, False)

            # Condition 1: Sample Threshold
            if samples < self.retrain_min_samples:
                return

            # Condition 2: Time Threshold (production stability)
            if last_retrain and (now - last_retrain < self.retrain_min_interval):
                return

            # Avoid concurrent evolution cycles for the same symbol
            if is_evolving:
                logger.debug(f"Learner: Evolution already in progress for {symbol}; skipping")
                return

            # Mark evolution in progress
            self._is_evolving[symbol] = True

        # Condition 3: Drift Threshold
        try:
            audit_report = self.audit.run_audit(symbol, "MULTITF", store=self.store)
            if not audit_report or "drift_score" not in audit_report:
                logger.error(
                    f"Learner: Audit failed or missing drift_score for {symbol}; skipping evolution"
                )
                return

            drift_score = float(audit_report.get("drift_score", 0.0))

            if drift_score > self.drift_threshold:
                logger.info(
                    f"Evolution triggered for {symbol}: Drift={drift_score:.4f}, Samples={samples}"
                )

                engines = self.engines
                if not engines:
                    logger.warning(
                        f"Learner: No engines configured for {symbol}; skipping evolution"
                    )
                    return

                # Option A: evolve all selected engines
                for engine in engines:
                    try:
                        result = self.trainer.train_with_experiments(
                            symbol=symbol,
                            timeframe=active_tfs,
                            model_type=engine,
                        )
                        logger.info(
                            f"Learner: Symbol {symbol} evolved engine '{engine}' "
                            f"to version {result.get('version_id')}"
                        )
                    except Exception as exc:
                        logger.error(
                            f"Learner: Evolution failed for {symbol} engine '{engine}': {exc}"
                        )

                with self._lock:
                    self._last_retrain_at[symbol] = now
                    self._new_samples_since_retrain[symbol] = 0

        except Exception as e:
            logger.error(f"Learner: Evolution cycle failed for {symbol}: {e}")

        finally:
            with self._lock:
                self._is_evolving[symbol] = False

    # ------------------------------------------------------------------
    # MANUAL / ORCHESTRATOR HOOKS
    # ------------------------------------------------------------------

    def force_retrain(
        self,
        symbol: str,
        features_by_symbol: Dict[str, pd.DataFrame],
    ) -> Dict[str, Any]:
        """
        Manual bypass for retraining.
        Evolves all selected engines for the symbol, regardless of drift.
        Returns a dict of {engine: result}.
        """
        active_tfs: List[str] = [
            tf for tf, feats in features_by_symbol.items()
            if feats is not None and not feats.empty
        ]
        if not active_tfs:
            logger.warning(f"Learner: No active timeframes for {symbol}; manual retrain skipped")
            return {}

        engines = self.engines
        if not engines:
            logger.warning(f"Learner: No engines configured for {symbol}; manual retrain skipped")
            return {}

        logger.info(f"Learner: Manual retrain forced for {symbol} on engines={engines}")

        results: Dict[str, Any] = {}
        for engine in engines:
            try:
                result = self.trainer.train_with_experiments(
                    symbol=symbol,
                    timeframe=active_tfs,
                    model_type=engine,
                )
                results[engine] = result
                logger.info(
                    f"Learner: Manual retrain for {symbol} engine '{engine}' "
                    f"produced version {result.get('version_id')}"
                )
            except Exception as exc:
                logger.error(
                    f"Learner: Manual retrain failed for {symbol} engine '{engine}': {exc}"
                )

        with self._lock:
            self._last_retrain_at[symbol] = dt.datetime.utcnow()
            self._new_samples_since_retrain[symbol] = 0
            self._is_evolving[symbol] = False

        return results

    def get_live_metrics(self, symbol: str) -> Dict[str, Any]:
        """Returns symbol-level health via MULTITF audit."""
        return self.audit.run_audit(symbol, "MULTITF", store=self.store)


# --- Global accessor for LearnerService (V4 cockpit) ---

_learner_service_global = None


def set_global_learner_service(service: LearnerService) -> None:
    """
    Register the process-wide LearnerService instance.
    """
    global _learner_service_global
    _learner_service_global = service


def get_global_learner_service() -> Optional[LearnerService]:
    """
    Return the process-wide LearnerService instance, or None if not set.
    """
    return _learner_service_global
