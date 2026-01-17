# mikebot/core/training_pipeline_v4.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from mikebot.core.triple_barrier import triple_barrier_labeling
from mikebot.features.feature_builder_v4 import FeatureBuilderV4
from mikebot.features.feature_pipeline import default_feature_pipeline
from mikebot.features.multi_tf_aggregator_v4 import MultiTFAggregatorV4, MultiTFConfigV4
from mikebot.experience.experience_store_v4 import ExperienceStoreV4


# ---------------------------------------------------------------------------
# Unified config (v4 cockpit + legacy knobs)
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfigV4:
    """
    Canonical training configuration for the unified pipeline.

    This replaces the old TrainPipeline config and the earlier v4 config,
    so everything that matters for training lives here.
    """

    # Experiment / bookkeeping
    experiment_name: str = "default_experiment"
    model_type: str = "custom"  # informational; model_factory is the authority
    random_state: int = 42

    # Symbol / timeframe (for logging + TB defaults)
    symbol: str = ""
    timeframe: str = ""

    # Core model hyperparameters (passed to model_factory if desired)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    # Target construction (triple-barrier)
    use_triple_barrier: bool = True
    tp_pips: int = 20
    sl_pips: int = 10
    max_horizon: int = 12
    pip_value: float = 0.0001
    drop_neutral: bool = False

    # Multi-timeframe enrichment
    multi_tf_timeframes: Optional[List[str]] = None  # e.g. ["5", "15", "60"]

    # Sample / regime weights
    sample_weights: Optional[pd.Series] = None
    regime_weights: Optional[Dict[str, float]] = None

    # Diagnostics toggles
    compute_regime_performance: bool = True
    compute_strategy_performance: bool = True
    compute_feature_importance: bool = True


# ---------------------------------------------------------------------------
# Unified training pipeline
# ---------------------------------------------------------------------------


class TrainingPipelineV4:
    """
    Unified training pipeline:

    - Builds features (v4 FeatureBuilderV4)
    - Optionally applies triple-barrier labeling
    - Optionally enriches with multi-TF features
    - Applies sample/regime weights
    - Trains model via model_factory
    - Computes metrics + diagnostics

    This is the single canonical pipeline; legacy TrainPipeline.py
    should be treated as deprecated once callers are migrated here.
    """

    def __init__(
        self,
        config: TrainingConfigV4,
        model_factory: Callable[[], Any],
        *,
        feature_builder: Optional[FeatureBuilderV4] = None,
        experience_store: Optional[ExperienceStoreV4] = None,
        multi_tf_aggregator: Optional[MultiTFAggregatorV4] = None,
    ) -> None:
        self.config = config
        self.model_factory = model_factory

        self.feature_builder = feature_builder or FeatureBuilderV4(
            pipeline=default_feature_pipeline()
        )
        self.experience_store = experience_store
        self.multi_tf_aggregator = multi_tf_aggregator

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    def train(
        self,
        *,
        candles: pd.DataFrame,
        target: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None,
        regimes: Optional[pd.DataFrame] = None,
        strategies: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a full unified training run.

        Parameters
        ----------
        candles:
            Raw OHLCV DataFrame (index is the canonical time index).
        target:
            Optional target Series aligned to candles.index.
            If None and use_triple_barrier=True, triple-barrier is applied.
        sample_weight:
            Optional per-sample weights aligned to candles.index.
        regimes:
            Optional regimes DataFrame with a 'regime_label' column.
        strategies:
            Optional dict of strategy DataFrames for diagnostics.

        Returns
        -------
        dict with keys:
            - model
            - features
            - target
            - metrics
            - predictions
        """
        # 1. Features
        X = self._build_features(candles=candles, strategies=strategies)

        # 2. Target
        y = self._build_target(candles=candles, X=X, target=target)

        # Align X to y after target construction
        X = X.reindex(y.index).dropna()
        y = y.reindex(X.index).dropna()

        # 3. Multi-TF enrichment (optional)
        X, y, regimes = self._maybe_enrich_multi_tf(
            candles=candles,
            X=X,
            y=y,
            regimes=regimes,
        )

        # 4. Weights
        weights = self._apply_weights(
            y=y,
            regimes=regimes,
            sample_weight=sample_weight,
        )

        # 5. Train model
        model = self._train_model(X=X, y=y, sample_weights=weights)

        # 6. Predictions + metrics
        metrics, pred_df = self._compute_metrics_and_diagnostics(
            model=model,
            X=X,
            y=y,
            regimes=regimes,
            strategies=strategies,
        )

        return {
            "model": model,
            "features": X,
            "target": y,
            "metrics": metrics,
            "predictions": pred_df,
        }

    # ------------------------------------------------------------------
    # 1. Feature building
    # ------------------------------------------------------------------

    def _build_features(
        self,
        *,
        candles: pd.DataFrame,
        strategies: Optional[Dict[str, pd.DataFrame]],
    ) -> pd.DataFrame:
        """
        v4-style feature building.

        The exact FeatureBuilderV4 API can be adjusted; this assumes a
        simple `build_features(candles, extra_context=...)` signature.
        """
        extra_context: Dict[str, Any] = {
            "symbol": self.config.symbol,
            "timeframe": self.config.timeframe,
            "strategies": strategies or {},
        }

        X = self.feature_builder.build_features(
            candles=candles,
            extra_context=extra_context,
        )
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        return X

    # ------------------------------------------------------------------
    # 2. Target construction
    # ------------------------------------------------------------------

    def _build_target(
        self,
        *,
        candles: pd.DataFrame,
        X: pd.DataFrame,
        target: Optional[pd.Series],
    ) -> pd.Series:
        # Explicit target path
        if target is not None:
            y = target.reindex(X.index)
            return y.dropna()

        # Triple-barrier path
        if not self.config.use_triple_barrier:
            raise ValueError(
                "No target provided and triple-barrier disabled in config."
            )

        tb = triple_barrier_labeling(
            df=candles,
            tp_pips=self.config.tp_pips,
            sl_pips=self.config.sl_pips,
            max_horizon=self.config.max_horizon,
            price_col="close",
            pip_value=self.config.pip_value,
        )
        y = pd.Series(tb, index=candles.index).reindex(X.index)

        if self.config.drop_neutral:
            y = y[y != 0]

        return y.dropna()

    # ------------------------------------------------------------------
    # 3. Multi-timeframe enrichment
    # ------------------------------------------------------------------

    def _maybe_enrich_multi_tf(
        self,
        *,
        candles: pd.DataFrame,
        X: pd.DataFrame,
        y: pd.Series,
        regimes: Optional[pd.DataFrame],
    ) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame]]:
        """
        Optional multi-TF enrichment.

        This is intentionally conservative: if anything goes wrong or
        produces an empty join, we fall back to the single-TF X/y.
        """
        if not self.config.multi_tf_timeframes:
            return X, y, regimes

        if self.multi_tf_aggregator is None:
            # Build a simple aggregator on the fly using numeric minutes
            targets: List[Tuple[str, int]] = []
            for tf in self.config.multi_tf_timeframes:
                try:
                    minutes = int(tf)
                except ValueError:
                    # If parsing fails, skip this TF
                    continue
                targets.append((tf, minutes))

            if not targets:
                return X, y, regimes

            self.multi_tf_aggregator = MultiTFAggregatorV4(
                config=MultiTFConfigV4(targets=targets)
            )

        # Offline aggregation over the full candle history
        tf_features = self.multi_tf_aggregator.aggregate_history(candles=candles)
        tf_features = {
            tf: df for tf, df in tf_features.items() if df is not None and not df.empty
        }

        if not tf_features:
            return X, y, regimes

        # For simplicity, we inner-join all TFs on the existing X index.
        # You can refine this alignment if you want more precise time handling.
        merged_multi_tf = None
        for tf_name, df in tf_features.items():
            df = df.copy()
            df.index = candles.index[: len(df)]
            df = df.reindex(X.index)
            df = df.add_prefix(f"tf_{tf_name}_")
            merged_multi_tf = df if merged_multi_tf is None else merged_multi_tf.join(
                df, how="inner"
            )

        if merged_multi_tf is None or merged_multi_tf.empty:
            return X, y, regimes

        X_joined = X.join(merged_multi_tf, how="inner")
        if X_joined.empty:
            return X, y, regimes

        new_index = X_joined.index
        y_aligned = y.reindex(new_index).dropna()
        X_final = X_joined.loc[y_aligned.index]

        regimes_final = None
        if regimes is not None and not regimes.empty:
            regimes_final = regimes.reindex(X_final.index)

        return X_final, y_aligned, regimes_final

    # ------------------------------------------------------------------
    # 4. Weights
    # ------------------------------------------------------------------

    def _apply_weights(
        self,
        *,
        y: pd.Series,
        regimes: Optional[pd.DataFrame],
        sample_weight: Optional[pd.Series],
    ) -> np.ndarray:
        n = len(y)
        w = np.ones(n, dtype=float)

        # Per-call sample_weight
        if sample_weight is not None:
            sw = sample_weight.reindex(y.index).fillna(0.0)
            w *= sw.values

        # Config-level sample_weights
        if self.config.sample_weights is not None:
            sw = self.config.sample_weights.reindex(y.index).fillna(0.0)
            w *= sw.values

        # Regime weights
        if self.config.regime_weights and regimes is not None and not regimes.empty:
            reg = regimes.reindex(y.index)
            if "regime_label" in reg.columns:
                labels = reg["regime_label"].astype(str)
                for regime_label, weight in self.config.regime_weights.items():
                    mask = labels == str(regime_label)
                    w[mask.values] *= float(weight)

        return w

    # ------------------------------------------------------------------
    # 5. Model training
    # ------------------------------------------------------------------

    def _train_model(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: np.ndarray,
    ) -> Any:
        model = self.model_factory()
        # Expect model to support sample_weight in fit; if not, caller
        # can wrap model_factory accordingly.
        model.fit(X.values, y.values, sample_weight=sample_weights)
        return model

    # ------------------------------------------------------------------
    # 6. Metrics + diagnostics
    # ------------------------------------------------------------------

    def _predict_scores(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X.values)
            if proba.shape[1] == 2:
                return proba[:, 1] - proba[:, 0]
            return np.argmax(proba, axis=1).astype(float)
        if hasattr(model, "decision_function"):
            return model.decision_function(X.values).astype(float)
        return model.predict(X.values).astype(float)

    def _compute_metrics_and_diagnostics(
        self,
        *,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        regimes: Optional[pd.DataFrame],
        strategies: Optional[Dict[str, pd.DataFrame]],
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        preds = self._predict_scores(model, X)
        pred_df = pd.DataFrame({"prediction": preds}, index=X.index)

        # Binary classification assumption: sign-based
        y_true = y.values
        y_pred_class = np.where(preds > 0, 1, 0)

        basic = self._compute_basic_metrics(y_true=y_true, y_pred=y_pred_class)
        extended = self._compute_extended_metrics(
            y=y, pred_df=pred_df, scores=preds, y_pred_class=y_pred_class
        )

        metrics: Dict[str, Any] = {**basic, **extended}

        if self.config.compute_feature_importance:
            metrics["feature_importance"] = self._compute_feature_importance(
                model=model,
                X=X,
            )

        if self.config.compute_regime_performance:
            metrics["regime_performance"] = self._compute_regime_performance(
                y_true=y_true,
                y_pred_class=y_pred_class,
                regimes=regimes,
                index=X.index,
            )

        if self.config.compute_strategy_performance:
            metrics["strategy_performance"] = self._compute_strategy_performance(
                y_true=y_true,
                y_pred_class=y_pred_class,
                strategies=strategies,
                index=X.index,
                X=X,
            )

        return metrics, pred_df

    # --- basic / extended metrics -------------------------------------

    @staticmethod
    def _compute_basic_metrics(
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        out["accuracy"] = float(accuracy_score(y_true, y_pred))

        # Guard against degenerate cases
        try:
            out["precision"] = float(
                precision_score(y_true, y_pred, zero_division=0)
            )
            out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
            out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        except Exception:
            out["precision"] = 0.0
            out["recall"] = 0.0
            out["f1"] = 0.0

        return out

    @staticmethod
    def _compute_extended_metrics(
        *,
        y: pd.Series,
        pred_df: pd.DataFrame,
        scores: np.ndarray,
        y_pred_class: np.ndarray,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        # AUC (if both classes present)
        try:
            if len(np.unique(y.values)) > 1:
                out["auc"] = float(roc_auc_score(y.values, scores))
        except Exception:
            out["auc"] = float("nan")

        # Confusion matrix
        try:
            tn, fp, fn, tp = confusion_matrix(
                y.values, y_pred_class, labels=[0, 1]
            ).ravel()
            out["tp"] = int(tp)
            out["tn"] = int(tn)
            out["fp"] = int(fp)
            out["fn"] = int(fn)
        except Exception:
            out["tp"] = out["tn"] = out["fp"] = out["fn"] = 0

        # You can add more extended metrics here as needed.
        return out

    # --- regime performance -------------------------------------------

    @staticmethod
    def _compute_regime_performance(
        *,
        y_true: np.ndarray,
        y_pred_class: np.ndarray,
        regimes: Optional[pd.DataFrame],
        index: pd.Index,
    ) -> Dict[str, Any]:
        if regimes is None or regimes.empty:
            return {}

        reg = regimes.copy().reindex(index)
        if "regime_label" not in reg.columns:
            return {}

        labels = reg["regime_label"].astype(str)
        performance: Dict[str, Any] = {}

        for regime_label in sorted(labels.dropna().unique()):
            mask = labels == regime_label
            if mask.sum() == 0:
                continue

            y_r = y_true[mask.values]
            y_pred_r = y_pred_class[mask.values]

            metrics_r = TrainingPipelineV4._compute_basic_metrics(
                y_true=y_r,
                y_pred=y_pred_r,
            )
            performance[str(regime_label)] = metrics_r

        return performance

    # --- strategy performance -----------------------------------------

    @staticmethod
    def _compute_strategy_performance(
        *,
        y_true: np.ndarray,
        y_pred_class: np.ndarray,
        strategies: Optional[Dict[str, pd.DataFrame]],
        index: pd.Index,
        X: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        if not strategies:
            return {}

        performance: Dict[str, Any] = {}

        for name, df in strategies.items():
            if df is None or df.empty:
                continue

            s = None
            if X is not None:
                feat_name = f"strategy_{name}_signal"
                if feat_name in X.columns:
                    s = X[feat_name].reindex(index).fillna(0.0)

            if s is None:
                if "signal" not in df.columns:
                    continue
                s = df["signal"].reindex(index).fillna(0.0)

            mask = s != 0
            if mask.sum() == 0:
                continue

            y_s = y_true[mask.values]
            strat_side = np.where(s[mask] > 0, 1, -1)
            y_signed = np.sign(y_s)

            win_rate = float((y_signed == strat_side).mean())

            performance[name] = {
                "win_rate": win_rate,
                "accuracy": win_rate,
                "directional_hit_rate": win_rate,
                "support": int(mask.sum()),
            }

        return performance

    # --- feature importance -------------------------------------------

    @staticmethod
    def _compute_feature_importance(
        *,
        model: Any,
        X: pd.DataFrame,
    ) -> Dict[str, float]:
        importances: Dict[str, float] = {}

        # Tree-based models with feature_importances_
        if hasattr(model, "feature_importances_"):
            raw = model.feature_importances_
            if len(raw) == X.shape[1]:
                for col, val in zip(X.columns, raw):
                    importances[str(col)] = float(val)

        # XGBoost-style models with booster.get_score
        elif hasattr(model, "get_booster"):
            try:
                booster = model.get_booster()
                score = booster.get_score(importance_type="gain")
                for k, v in score.items():
                    if k.startswith("f"):
                        idx = int(k[1:])
                        if idx < len(X.columns):
                            importances[str(X.columns[idx])] = float(v)
                if not importances and hasattr(model, "feature_names_in_"):
                    for col in model.feature_names_in_:
                        importances[str(col)] = float(0.0)
            except Exception:
                pass

        total = sum(importances.values())
        if total > 0:
            for k in list(importances.keys()):
                importances[k] = importances[k] / total

        return importances