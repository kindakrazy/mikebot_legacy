from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence, Union, Callable

import datetime as dt

import numpy as np
import pandas as pd

from mikebot.core.experience_store import ExperienceStore
from mikebot.core.self_audit import SelfAudit
from mikebot.core.model_lineage import ModelLineageRegistry
from mikebot.core.model_registry import ModelRegistry
from mikebot.core.experiment_record import ExperimentRecord
from mikebot.strategies.strategy_registry import StrategyRegistry
from mikebot.core.unified_training_pipeline import (
    UnifiedTrainingPipeline,
    UnifiedTrainingConfig,
)


class MetaTrainerV4:
    """
    MetaTrainer wired to UnifiedTrainingPipeline, using legacy ExperienceStore.

    Responsibilities:
    - load experience for (symbol, timeframes) via ExperienceStore
    - generate experiment configurations
    - run experiments via UnifiedTrainingPipeline (feature mode, X/y)
    - audit results via SelfAudit
    - compare experiments and choose the best
    - record experiments in ModelLineageRegistry
    - record promotions in ModelLineageRegistry
    - update ModelRegistry with symbol-level active version
    - return summaries for UI/logs
    """

    def __init__(
        self,
        experience_store: ExperienceStore,
        lineage: ModelLineageRegistry,
        registry: ModelRegistry,
        strategy_registry: StrategyRegistry,
        model_factories: Dict[str, Callable[[], Any]],
        model_dir: Optional[Path] = None,
    ) -> None:
        self.experience_store = experience_store
        self.lineage = lineage
        self.registry = registry
        self.strategy_registry = strategy_registry
        self.model_factories = model_factories
        self.model_dir = model_dir

        self.audit = SelfAudit()

    # ------------------------------------------------------------------ #
    # MAIN ENTRYPOINT (multi-TF aware)                                   #
    # ------------------------------------------------------------------ #

    def train_with_experiments(
        self,
        symbol: str,
        timeframe: Union[str, Sequence[str]],
        model_type: str,
    ) -> Dict[str, Any]:
        tfs = self._normalize_timeframes(timeframe)

        # Load experience (MULTITF synthetic view if multiple TFs)
        if len(tfs) > 1:
            experience = self.experience_store.load_all(symbol, "MULTITF")
        else:
            experience = self.experience_store.load_all(symbol, tfs[0])

        # Require labels
        labels = experience.get("labels")
        if labels is None or labels.empty:
            raise RuntimeError(f"MetaTrainerV4: no labels available for {symbol} {tfs}")

        # Generate experiments
        experiments = self.generate_experiments(symbol, tfs, experience)
        if not experiments:
            raise RuntimeError(f"MetaTrainerV4: no experiments generated for {symbol} {tfs}")

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

        # Promote best model
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
                    timeframe="MULTITF" if len(tfs) > 1 else tfs[0],
                    model_type=model_type,
                    predictions=preds,
                )
            except Exception:
                pass

        if errs is not None and hasattr(self.experience_store, "save_errors"):
            try:
                self.experience_store.save_errors(
                    symbol=symbol,
                    timeframe="MULTITF" if len(tfs) > 1 else tfs[0],
                    model_type=model_type,
                    errors=errs,
                )
            except Exception:
                pass

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

    # ------------------------------------------------------------------ #
    # EXPERIMENT GENERATION                                              #
    # ------------------------------------------------------------------ #

    def generate_experiments(
        self,
        symbol: str,
        timeframes: Sequence[str],
        experience: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        rep_features = experience.get("features")
        rep_labels = experience.get("labels")
        rep_predictions = experience.get("predictions")
        rep_errors = experience.get("errors")
        rep_strategies = experience.get("strategies", {}) or {}
        rep_regimes = experience.get("regimes")

        if rep_features is None or rep_labels is None:
            rep_features = rep_features or pd.DataFrame()
            rep_labels = rep_labels or pd.DataFrame()

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

    # ------------------------------------------------------------------ #
    # RUN EXPERIMENT (feature mode, X/y)                                 #
    # ------------------------------------------------------------------ #

    def run_experiment(
        self,
        symbol: str,
        timeframes: Sequence[str],
        model_type: str,
        config: Dict[str, Any],
        experience: Dict[str, Any],
    ) -> Dict[str, Any]:
        features: Optional[pd.DataFrame] = experience.get("features")
        labels: Optional[pd.DataFrame] = experience.get("labels")
        regimes: Optional[pd.DataFrame] = experience.get("regimes")
        strategies: Dict[str, pd.DataFrame] = experience.get("strategies", {}) or {}

        if features is None or labels is None:
            raise RuntimeError("MetaTrainerV4.run_experiment: missing features or labels")

        if isinstance(labels, pd.DataFrame) and labels.shape[1] == 1:
            labels_series = labels.iloc[:, 0]
        elif isinstance(labels, pd.DataFrame):
            labels_series = labels.iloc[:, 0]
        else:
            labels_series = labels

        sample_weights = config.get("sample_weights")
        strategy_features = config.get("strategy_features")
        regime_weights = config.get("regime_weights")
        hyperparameters = config.get("hyperparameters") or {}

        mapped: Dict[str, Any] = {}

        if "rf_n_estimators" in hyperparameters:
            mapped["n_estimators"] = int(hyperparameters["rf_n_estimators"])
        if "rf_max_depth" in hyperparameters:
            mapped["max_depth"] = int(hyperparameters["rf_max_depth"])

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

        for k, v in hyperparameters.items():
            if k not in mapped:
                mapped[k] = v

        hyperparameters = mapped

        training_config = UnifiedTrainingConfig(
            experiment_name=f"{symbol}_MULTITF_{model_type}",
            model_type=model_type,
            hyperparameters=hyperparameters,
            symbol=symbol,
            timeframe="MULTITF" if len(timeframes) > 1 else timeframes[0],
            use_triple_barrier=False,
            multi_tf_timeframes=None,
            sample_weights=None,
            regime_weights=regime_weights,
            compute_regime_performance=True,
            compute_strategy_performance=True,
            compute_feature_importance=True,
        )

        if model_type not in self.model_factories:
            raise RuntimeError(f"MetaTrainerV4: no model_factory for model_type={model_type}")
        model_factory = self.model_factories[model_type]

        pipeline = UnifiedTrainingPipeline(
            config=training_config,
            model_factory=model_factory,
        )

        sw_series = None
        if sample_weights is not None:
            sw_series = pd.Series(sample_weights, index=labels_series.index)

        out = pipeline.train(
            X=features,
            y=labels_series,
            sample_weight=sw_series,
            regimes=regimes,
            strategies=strategy_features if strategy_features is not None else strategies,
        )

        predictions: pd.DataFrame = out["predictions"]
        metrics: Dict[str, Any] = out["metrics"]

        global_profile = self.audit.compute_error_profile(
            labels, predictions,
        )

        result = {
            "experiment_type": config.get("experiment_type"),
            "description": config.get("description"),
            "metrics": metrics,
            "global_error_profile": global_profile,
            "regime_performance": metrics.get("regime_performance", {}),
            "strategy_performance": metrics.get("strategy_performance", {}),
            "training_config": training_config,
            "model_path": None,
            "metrics_path": None,
            "predictions": predictions,
            "errors": None,
        }

        return result

    # ------------------------------------------------------------------ #
    # COMPARE EXPERIMENTS                                                #
    # ------------------------------------------------------------------ #

    def compare_experiments(
        self,
        results: List[Dict[str, Any]],
        primary_metric: str = "accuracy",
    ) -> Dict[str, Any]:
        if not results:
            raise RuntimeError("MetaTrainerV4.compare_experiments: no results provided")

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

    # ------------------------------------------------------------------ #
    # PROMOTE BEST MODEL                                                 #
    # ------------------------------------------------------------------ #

    def promote_best_model(
        self,
        symbol: str,
        timeframes: Sequence[str],
        model_type: str,
        best_result: Dict[str, Any],
    ) -> str:
        metrics: Dict[str, Any] = best_result["metrics"]
        training_config: UnifiedTrainingConfig = best_result["training_config"]

        try:
            latest = self.lineage.get_latest(symbol, "MULTITF", model_type)
            parent_version = latest[0] if latest is not None else None
        except Exception:
            parent_version = None

        model_path = Path(self.model_dir or ".") / f"{symbol}_MULTITF_{model_type}.bin"
        metrics_path = Path(self.model_dir or ".") / f"{symbol}_MULTITF_{model_type}_metrics.json"
        model_tag = f"{symbol}_MULTITF_{model_type}"

        record = ExperimentRecord.from_pipeline_outputs(
            symbol=symbol,
            timeframe="MULTITF",
            model_type=model_type,
            version_id="",
            parent_version_id=parent_version,
            metrics=metrics,
            experiment_type=training_config.experiment_name,
            notes="",
            model_path=model_path,
            metrics_path=metrics_path,
            model_tag=model_tag,
        )

        version_id = self.lineage.record_experiment(record, set_best=True)

        try:
            self.lineage.record_promotion(
                symbol=symbol,
                timeframe="MULTITF",
                model_type=model_type,
                version_id=version_id,
                reason=f"Promoted by MetaTrainerV4 at {dt.datetime.utcnow().isoformat()}",
            )
        except Exception:
            pass

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
                            f"{training_config.experiment_name} experiment; "
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

    # ------------------------------------------------------------------ #
    # SUMMARIES                                                          #
    # ------------------------------------------------------------------ #

    def summaries(
        self,
        experiment_results: List[Dict[str, Any]],
        best_result: Dict[str, Any],
    ) -> Dict[str, Any]:
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

    # ------------------------------------------------------------------ #
    # INTERNAL HELPERS                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_timeframes(timeframe: Union[str, Sequence[str]]) -> List[str]:
        if isinstance(timeframe, str):
            return [timeframe]
        return list(timeframe)

    @staticmethod
    def _build_metrics_summary(metrics: Dict[str, Any]) -> Dict[str, Any]:
        keys = ["accuracy", "win_rate", "mean_abs_error", "rmse", "directional_hit_rate"]
        return {k: metrics.get(k) for k in keys if k in metrics}

    def _default_hyperparams(
        self,
        symbol: str,
        timeframes: Sequence[str],
    ) -> Dict[str, Any]:
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
        base = self._default_hyperparams(symbol, timeframes)
        variants: List[Dict[str, Any]] = []

        v1 = dict(base)
        v1["rf_max_depth"] = base["rf_max_depth"] + 2
        variants.append(v1)

        v2 = dict(base)
        v2["rf_n_estimators"] = base["rf_n_estimators"] + 100
        variants.append(v2)

        v3 = dict(base)
        v3["xgb_max_depth"] = base["xgb_max_depth"] + 2
        variants.append(v3)

        v4 = dict(base)
        v4["xgb_learning_rate"] = max(0.01, base["xgb_learning_rate"] * 0.5)
        variants.append(v4)

        return variants

    @staticmethod
    def _build_error_focused_weights(errors: pd.DataFrame) -> Optional[np.ndarray]:
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
        if labels is None or labels.empty:
            return None

        n = len(labels)
        if n <= 0:
            return None

        idx = np.arange(1, n + 1, dtype=float)
        w = 1.0 + (idx / n)
        return w