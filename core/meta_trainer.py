# mikebot/core/meta_trainer.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence, Tuple, Union

import datetime as dt

import numpy as np
import pandas as pd

from mikebot.core.experience_store import ExperienceStore
from mikebot.core.self_audit import SelfAudit
from mikebot.core.model_lineage import ModelLineageRegistry
from mikebot.core.model_registry import ModelRegistry
from mikebot.core.train_pipeline import TrainPipeline, TrainingConfig
from mikebot.core.experiment_record import ExperimentRecord
from mikebot.strategies.strategy_registry import StrategyRegistry


class MetaTrainer:
    """
    The Model Scientist (MULTITF / symbol-level aware).

    Responsibilities:
    - load experience for (symbol, timeframes)
    - generate experiment configurations
    - run experiments via TrainPipeline (symbol-level MULTITF)
    - audit results via SelfAudit
    - compare experiments and choose the best
    - record experiments in ModelLineageRegistry (v4)
    - record promotions in ModelLineageRegistry (v4)
    - update ModelRegistry with symbol-level active version
    - return summaries for UI/logs
    """

    def __init__(
        self,
        experience_store: ExperienceStore,
        lineage: ModelLineageRegistry,
        registry: ModelRegistry,
        strategy_registry: StrategyRegistry,
        train_pipeline: TrainPipeline,
        model_dir: Optional[Path] = None,
    ) -> None:
        self.experience_store = experience_store
        self.lineage = lineage
        self.registry = registry
        self.strategy_registry = strategy_registry
        self.train_pipeline = train_pipeline
        self.model_dir = model_dir

        self.audit = SelfAudit()

    # ------------------------------------------------------------------
    # MAIN ENTRYPOINT (multi-TF aware)
    # ------------------------------------------------------------------

    def train_with_experiments(
        self,
        symbol: str,
        timeframe: Union[str, Sequence[str]],
        model_type: str,
    ) -> Dict[str, Any]:
        """
        Main entrypoint for training with experiments.

        timeframe may be a single TF string (e.g., "M5") or a sequence
        (e.g., ["M1","M5","M15"]). Internally we treat training as symbol-level
        and use timeframe list `tfs`. For lineage/registry we use "MULTITF".
        """
        tfs = self._normalize_timeframes(timeframe)

        # Load experience
        if len(tfs) > 1 and hasattr(self.experience_store, "load_multi_tf"):
            try:
                multi = self.experience_store.load_multi_tf(symbol, tfs)
                experience = {"multi": multi}
            except Exception:
                experience = self.experience_store.load_all(symbol, tfs[0])
        else:
            experience = self.experience_store.load_all(symbol, tfs[0])

        # Require labels
        has_labels = False
        if "multi" in experience:
            for tf, data in experience["multi"].items():
                if data and data.get("labels") is not None:
                    has_labels = True
                    break
        else:
            has_labels = experience.get("labels") is not None

        if not has_labels:
            raise RuntimeError(f"MetaTrainer: no labels available for {symbol} {tfs}")

        # Generate experiments
        experiments = self.generate_experiments(symbol, tfs, experience)
        if not experiments:
            raise RuntimeError(f"MetaTrainer: no experiments generated for {symbol} {tfs}")

        # Run experiments
        experiment_results: List[Dict[str, Any]] = []
        for config in experiments:
            result = self.run_experiment(
                symbol=symbol,
                timeframes=tfs,
                model_type=model_type,
                config=config,
                experience=experience,
            )
            experiment_results.append(result)

        # Audit + pick best
        best_result = self.compare_experiments(experiment_results)

        # Promote best model (v4 lineage + registry)
        version_id = self.promote_best_model(
            symbol=symbol,
            timeframes=tfs,
            model_type=model_type,
            best_result=best_result,
        )

        # Save predictions/errors back to ExperienceStore
        preds = best_result.get("predictions")
        errs = best_result.get("errors")

        if preds is not None and hasattr(self.experience_store, "save_predictions"):
            try:
                self.experience_store.save_predictions(
                    symbol=symbol,
                    timeframe="MULTITF",
                    model_type=model_type,
                    predictions=preds,
                )
            except Exception:
                pass

        if errs is not None and hasattr(self.experience_store, "save_errors"):
            try:
                self.experience_store.save_errors(
                    symbol=symbol,
                    timeframe="MULTITF",
                    model_type=model_type,
                    errors=errs,
                )
            except Exception:
                pass

        # Summaries
        summary = self.summaries(experiment_results, best_result)
        summary["version_id"] = version_id
        summary["symbol"] = symbol
        summary["timeframes"] = tfs
        summary["model_type"] = model_type

        return {
            "version_id": version_id,
            "best_experiment": best_result,
            "all_experiments": experiment_results,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # EXPERIMENT GENERATION
    # ------------------------------------------------------------------

    def generate_experiments(
        self,
        symbol: str,
        timeframes: Sequence[str],
        experience: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Generate experiment configs for this symbol across requested timeframes.
        """
        # Representative TF for audit
        rep_features = None
        rep_labels = None
        rep_predictions = None
        rep_errors = None
        rep_strategies = {}
        rep_regimes = None

        if "multi" in experience:
            for tf in timeframes:
                tf_data = experience["multi"].get(tf)
                if not tf_data:
                    continue
                if tf_data.get("features") is not None:
                    rep_features = tf_data.get("features")
                    rep_labels = tf_data.get("labels")
                    rep_predictions = tf_data.get("predictions")
                    rep_errors = tf_data.get("errors")
                    rep_strategies = tf_data.get("strategies", {}) or {}
                    rep_regimes = tf_data.get("regimes")
                    break
        else:
            rep_features = experience.get("features")
            rep_labels = experience.get("labels")
            rep_predictions = experience.get("predictions")
            rep_errors = experience.get("errors")
            rep_strategies = experience.get("strategies", {}) or {}
            rep_regimes = experience.get("regimes")

        if rep_features is None or rep_labels is None:
            rep_features = rep_features or pd.DataFrame()
            rep_labels = rep_labels or pd.DataFrame()

        # Global audit
        if rep_predictions is not None and not rep_predictions.empty:
            global_profile = self.audit.compute_error_profile(rep_labels, rep_predictions)
        else:
            global_profile = {
                "n_samples": len(rep_labels) if rep_labels is not None else 0,
                "accuracy": np.nan,
                "mean_error": np.nan,
                "mean_abs_error": np.nan,
                "rmse": np.nan,
                "directional_hit_rate": np.nan,
            }

        regime_perf = (
            self.audit.evaluate_regime_performance(rep_regimes, rep_labels, rep_predictions)
            if rep_regimes is not None and rep_predictions is not None
            else {}
        )
        strategy_perf = (
            self.audit.evaluate_strategy_performance(rep_strategies, rep_labels)
            if rep_strategies
            else {}
        )

        recommendations = self.audit.recommend_training_adjustments(
            global_error_profile=global_profile,
            regime_performance=regime_perf,
            strategy_performance=strategy_perf,
        )

        experiments: List[Dict[str, Any]] = []

        # Baseline
        experiments.append(
            {
                "experiment_type": "baseline",
                "description": "baseline training on all data",
                "sample_weights": None,
                "strategy_features": None,
                "regime_weights": None,
                "hyperparameters": self._default_hyperparams(symbol, timeframes),
            }
        )

        # Error-focused
        if recommendations.get("use_error_focused_experiment") and rep_errors is not None:
            sample_weights = self._build_error_focused_weights(rep_errors)
            experiments.append(
                {
                    "experiment_type": "error_focused",
                    "description": "oversample hard examples based on error magnitude",
                    "sample_weights": sample_weights,
                    "strategy_features": None,
                    "regime_weights": None,
                    "hyperparameters": self._default_hyperparams(symbol, timeframes),
                }
            )

        # Recency-weighted
        if recommendations.get("use_recency_weighted_experiment"):
            sample_weights = self._build_recency_weights(rep_labels)
            experiments.append(
                {
                    "experiment_type": "recency_weighted",
                    "description": "weight recent samples more heavily",
                    "sample_weights": sample_weights,
                    "strategy_features": None,
                    "regime_weights": None,
                    "hyperparameters": self._default_hyperparams(symbol, timeframes),
                }
            )

        # Strategy-augmented
        if rep_strategies:
            experiments.append(
                {
                    "experiment_type": "strategy_augmented",
                    "description": "include strategy signals as features",
                    "sample_weights": None,
                    "strategy_features": rep_strategies,
                    "regime_weights": recommendations.get("regime_weights") or None,
                    "hyperparameters": self._default_hyperparams(symbol, timeframes),
                }
            )

        # Hyperparameter variants
        hp_variants = self._hyperparam_variants(symbol, timeframes)
        for hp in hp_variants:
            experiments.append(
                {
                    "experiment_type": "hyperparam_variant",
                    "description": f"hyperparam variant: {hp}",
                    "sample_weights": None,
                    "strategy_features": None,
                    "regime_weights": None,
                    "hyperparameters": hp,
                }
            )

        return experiments

    # ------------------------------------------------------------------
    # RUN EXPERIMENT (multi-TF aware)
    # ------------------------------------------------------------------

    def run_experiment(
        self,
        symbol: str,
        timeframes: Sequence[str],
        model_type: str,
        config: Dict[str, Any],
        experience: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Train a model using TrainPipeline with the given experiment config.
        """
        features: Optional[pd.DataFrame] = None
        labels: Optional[pd.DataFrame] = None
        regimes: Optional[pd.DataFrame] = None
        strategies: Dict[str, pd.DataFrame] = {}

        if "multi" in experience:
            base_tf = timeframes[0] if timeframes else next(iter(experience["multi"].keys()))
            tf_data = experience["multi"].get(base_tf) or {}
            features = tf_data.get("features")
            labels = tf_data.get("labels")
            regimes = tf_data.get("regimes")
            strategies = tf_data.get("strategies", {}) or {}
        else:
            features = experience.get("features")
            labels = experience.get("labels")
            regimes = experience.get("regimes")
            strategies = experience.get("strategies", {}) or {}

        sample_weights = config.get("sample_weights")
        strategy_features = config.get("strategy_features")
        regime_weights = config.get("regime_weights")
        hyperparameters = config.get("hyperparameters") or {}

        # --- v4 hyperparameter mapping layer ---
        mapped: Dict[str, Any] = {}

        # RF mappings
        if "rf_n_estimators" in hyperparameters:
            mapped["n_estimators"] = int(hyperparameters["rf_n_estimators"])
        if "rf_max_depth" in hyperparameters:
            mapped["max_depth"] = int(hyperparameters["rf_max_depth"])

        # XGB mappings
        if "xgb_n_estimators" in hyperparameters:
            mapped["n_estimators"] = int(hyperparameters["xgb_n_estimators"])
        if "xgb_max_depth" in hyperparameters:
            mapped["max_depth"] = int(hyperparameters["xgb_max_depth"])
        if "xgb_learning_rate" in hyperparameters:
            mapped["learning_rate"] = float(hyperparameters["xgb_learning_rate"])
        if "xgb_subsample" in hyperparameters:
            mapped["subsample"] = float(hyperparameters["xgb_subsample"])
        if "xgb_colsample_bytree" in hyperparameters:
            mapped["colsample_bytree"] = float(hyperparameters["xgb_colsample_bytree"])

        # Merge with any generic keys already present
        for k, v in hyperparameters.items():
            if k not in mapped:
                mapped[k] = v

        hyperparameters = mapped
        # --- end mapping layer ---

        training_config = TrainingConfig(
            hyperparameters=hyperparameters,
            sample_weights=sample_weights,
            strategy_features=None,
            regime_weights=regime_weights,
            warm_start_model=None,
            experiment_type=config.get("experiment_type", "baseline"),
            notes=config.get("description"),
        )

        # Train MULTITF model
        train_result = self.train_pipeline.train_multitf(
            symbol=symbol,
            timeframes=list(timeframes),
            model_type=model_type,
            features=features,
            labels=labels,
            regimes=regimes,
            strategies=strategy_features if strategy_features is not None else strategies,
            config=training_config,
        )

        predictions: pd.DataFrame = train_result["predictions"]
        errors: pd.DataFrame = train_result.get("errors")
        metrics: Dict[str, Any] = train_result["metrics"]
        regime_perf: Dict[str, Any] = train_result.get("regime_performance", {})
        strategy_perf: Dict[str, Any] = train_result.get("strategy_performance", {})
        model_path: Path = train_result["model_path"]
        metrics_path: Path = train_result["metrics_path"]

        # Global profile
        global_profile = self.audit.compute_error_profile(
            labels if labels is not None else pd.DataFrame(),
            predictions,
        )

        result = {
            "experiment_type": config.get("experiment_type"),
            "description": config.get("description"),
            "metrics": metrics,
            "global_error_profile": global_profile,
            "regime_performance": regime_perf,
            "strategy_performance": strategy_perf,
            "training_config": training_config,
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
            "predictions": predictions,
            "errors": errors,
        }

        return result

    # ------------------------------------------------------------------
    # COMPARE EXPERIMENTS
    # ------------------------------------------------------------------

    def compare_experiments(
        self,
        results: List[Dict[str, Any]],
        primary_metric: str = "accuracy",
    ) -> Dict[str, Any]:
        """
        Compare experiment results and return the best one.
        """
        if not results:
            raise RuntimeError("MetaTrainer.compare_experiments: no results provided")

        best: Optional[Dict[str, Any]] = None
        best_score: float = float("-inf")
        best_secondary: float = float("-inf")

        for res in results:
            metrics = res.get("metrics", {})
            g = res.get("global_error_profile", {})
            primary = metrics.get(primary_metric)
            if primary is None:
                continue

            secondary = g.get("directional_hit_rate") or 0.0

            score = float(primary)
            sec = float(secondary)

            if score > best_score or (score == best_score and sec > best_secondary):
                best = res
                best_score = score
                best_secondary = sec

        if best is None:
            best = results[0]

        return best

    # ------------------------------------------------------------------
    # PROMOTE BEST MODEL (v4 lineage + registry)
    # ------------------------------------------------------------------

    def promote_best_model(
        self,
        symbol: str,
        timeframes: Sequence[str],
        model_type: str,
        best_result: Dict[str, Any],
    ) -> str:
        """
        Update ModelLineageRegistry and ModelRegistry with the new best version.
        """
        metrics: Dict[str, Any] = best_result["metrics"]
        regime_perf: Dict[str, Any] = best_result.get("regime_performance", {})
        strategy_perf: Dict[str, Any] = best_result.get("strategy_performance", {})
        training_config: TrainingConfig = best_result["training_config"]

        # Determine parent version
        try:
            latest = self.lineage.get_latest(symbol, "MULTITF", model_type)
            parent_version = latest[0] if latest is not None else None
        except Exception:
            parent_version = None

        # Extract model_tag from filename
        model_path = Path(best_result["model_path"])
        metrics_path = Path(best_result["metrics_path"])
        model_tag = model_path.stem.split("_")[-1]

        # Build ExperimentRecord
        record = ExperimentRecord.from_pipeline_outputs(
            symbol=symbol,
            timeframe="MULTITF",
            model_type=model_type,
            version_id="",  # let lineage assign sequential vN
            parent_version_id=parent_version,
            metrics=metrics,
            experiment_type=training_config.experiment_type,
            notes=training_config.notes or "",
            model_path=model_path,
            metrics_path=metrics_path,
            model_tag=model_tag,
        )

        # Record experiment in lineage
        version_id = self.lineage.record_experiment(record, set_best=True)

        # Record promotion event
        try:
            self.lineage.record_promotion(
                symbol=symbol,
                timeframe="MULTITF",
                model_type=model_type,
                version_id=version_id,
                reason=f"Promoted by MetaTrainer at {dt.datetime.utcnow().isoformat()}",
            )
        except Exception:
            pass

        # Update ModelRegistry
        metrics_summary = self._build_metrics_summary(metrics)

        try:
            if hasattr(self.registry, "register_model"):
                try:
                    self.registry.register_model(
                        symbol=symbol,
                        timeframe="MULTITF",
                        model_type=model_type,
                        version=version_id,
                        model_path=model_path,
                        metrics_path=metrics_path,
                        metrics_summary=metrics_summary,
                        set_current=True,
                        set_best=True,
                        notes=(
                            f"{training_config.experiment_type} experiment; "
                            f"version {version_id} promoted as current/best"
                        ),
                    )
                except Exception:
                    self.registry.update_active_version(symbol, model_type, version_id)
            else:
                self.registry.update_active_version(symbol, model_type, version_id)
        except Exception:
            pass

        return version_id

    # ------------------------------------------------------------------
    # SUMMARIES
    # ------------------------------------------------------------------

    def summaries(
        self,
        experiment_results: List[Dict[str, Any]],
        best_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Produce human-readable summaries for logs or UI.
        """
        overview: List[Dict[str, Any]] = []
        for res in experiment_results:
            metrics = res.get("metrics", {})
            g = res.get("global_error_profile", {})
            overview.append(
                {
                    "experiment_type": res.get("experiment_type"),
                    "description": res.get("description"),
                    "accuracy": metrics.get("accuracy"),
                    "win_rate": metrics.get("win_rate"),
                    "directional_hit_rate": g.get("directional_hit_rate"),
                }
            )

        return {
            "best_experiment_type": best_result.get("experiment_type"),
            "best_metrics": best_result.get("metrics", {}),
            "experiments_overview": overview,
        }

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_timeframes(timeframe: Union[str, Sequence[str]]) -> List[str]:
        if isinstance(timeframe, str):
            return [timeframe]
        return list(timeframe)

    @staticmethod
    def _build_metrics_summary(metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compact summary for ModelRegistry.
        """
        keys = ["accuracy", "win_rate", "mean_abs_error", "rmse", "directional_hit_rate"]
        return {k: metrics.get(k) for k in keys if k in metrics}

    def _default_hyperparams(
        self,
        symbol: str,
        timeframes: Sequence[str],
    ) -> Dict[str, Any]:
        """
        Default hyperparameters for baseline experiments.
        """
        return {
            "rf_n_estimators": 200,
            "rf_max_depth": 8,
            "xgb_n_estimators": 300,
            "xgb_max_depth": 6,
            "xgb_learning_rate": 0.05,
            "xgb_subsample": 0.8,
            "xgb_colsample_bytree": 0.8,
        }

    def _hyperparam_variants(
        self,
        symbol: str,
        timeframes: Sequence[str],
    ) -> List[Dict[str, Any]]:
        """
        Simple deterministic hyperparameter variants.
        """
        base = self._default_hyperparams(symbol, timeframes)
        variants: List[Dict[str, Any]] = []

        # Slightly deeper RF
        v1 = dict(base)
        v1["rf_max_depth"] = base["rf_max_depth"] + 2
        variants.append(v1)

        # More RF trees
        v2 = dict(base)
        v2["rf_n_estimators"] = base["rf_n_estimators"] + 100
        variants.append(v2)

        # Deeper XGB
        v3 = dict(base)
        v3["xgb_max_depth"] = base["xgb_max_depth"] + 2
        variants.append(v3)

        # Slower learning rate
        v4 = dict(base)
        v4["xgb_learning_rate"] = max(0.01, base["xgb_learning_rate"] * 0.5)
        variants.append(v4)

        return variants

    @staticmethod
    def _build_error_focused_weights(errors: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Build sample weights proportional to absolute error.
        """
        if errors is None or errors.empty:
            return None

        numeric_cols = errors.select_dtypes(include=["number"]).columns
        if len(numeric_cols) == 0:
            return None

        e = errors[numeric_cols[0]].abs().values.astype(float)
        if e.size == 0:
            return None

        max_e = e.max()
        if max_e <= 0:
            return np.ones_like(e)

        e_norm = e / max_e
        return 1.0 + e_norm

    @staticmethod
    def _build_recency_weights(labels: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Build linearly increasing weights over time (more recent = higher weight).
        """
        if labels is None or labels.empty:
            return None

        n = len(labels)
        if n <= 0:
            return None

        # Oldest gets weight 1.0, newest gets weight 2.0
        idx = np.arange(1, n + 1, dtype=float)
        w = 1.0 + (idx / n)
        return w
