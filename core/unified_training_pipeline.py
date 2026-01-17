from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple, Callable

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


@dataclass
class UnifiedTrainingConfig:
    experiment_name: str = "default_experiment"
    model_type: str = "generic"

    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    symbol: str = ""
    timeframe: str = ""

    use_triple_barrier: bool = True
    tp_pips: float = 20.0
    sl_pips: float = 10.0
    max_horizon: int = 12
    pip_value: float = 0.0001
    drop_neutral: bool = False

    multi_tf_timeframes: Optional[List[str]] = None

    sample_weights: Optional[pd.Series] = None
    regime_weights: Optional[Dict[str, float]] = None

    compute_regime_performance: bool = True
    compute_strategy_performance: bool = True
    compute_feature_importance: bool = True


class UnifiedTrainingPipeline:
    """
    Canonical training pipeline.

    Two modes:
    - Candle mode: candles provided, pipeline builds features.
    - Feature mode: X/y provided, pipeline skips feature building entirely.
    """

    def __init__(
        self,
        config: UnifiedTrainingConfig,
        model_factory: Callable[[], Any],
    ) -> None:
        self.config = config
        self.model_factory = model_factory
        self.feature_builder = None

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def train(
        self,
        candles: Optional[pd.DataFrame] = None,
        target: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None,
        regimes: Optional[pd.DataFrame] = None,
        strategies: Optional[Dict[str, pd.DataFrame]] = None,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Train a model.

        Feature mode:
            - X (features) and y (target) provided
            - No feature building, no triple-barrier, no multi-TF enrichment

        Candle mode:
            - candles provided
            - pipeline builds features, constructs target if needed,
              and optionally enriches with multi-TF.
        """
        feature_mode = X is not None

        if feature_mode:
            # Feature mode: X/y are authoritative
            if y is None and target is not None:
                y = target
            if X is None or y is None:
                raise ValueError("Feature mode requires X and y (or target).")

            X = X.copy()
            y = y.reindex(X.index).dropna()
            X = X.reindex(y.index).dropna()
        else:
            if candles is None:
                raise ValueError("Candle mode requires candles.")
            # 1) Build features
            X = self._build_features(candles, strategies)
            # 2) Build target
            y = self._build_target(candles, X, target)
            X = X.reindex(y.index).dropna()
            y = y.reindex(X.index).dropna()
            # 3) Multi-TF enrichment
            X, y, regimes = self._maybe_enrich_multi_tf(candles, X, y, regimes)

        # 4) Weights
        weights = self._apply_weights(y, regimes, sample_weight)

        # 5) Train model
        model = self._train_model(X, y, weights)

        # 6) Metrics + diagnostics
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

    # ------------------------------------------------------------------ #
    # Feature construction (candle mode)                                 #
    # ------------------------------------------------------------------ #

    def _build_features(
        self,
        candles: pd.DataFrame,
        strategies: Optional[Dict[str, pd.DataFrame]],
    ) -> pd.DataFrame:
        extra_context = {
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

    # ------------------------------------------------------------------ #
    # Target construction (candle mode)                                  #
    # ------------------------------------------------------------------ #

    def _build_target(
        self,
        candles: pd.DataFrame,
        X: pd.DataFrame,
        target: Optional[pd.Series],
    ) -> pd.Series:
        if target is not None:
            y = target.reindex(X.index).dropna()
            return y.astype(float)

        if not self.config.use_triple_barrier:
            raise ValueError("No target provided and triple-barrier disabled.")

        tb = triple_barrier_labeling(
            df=candles,
            tp_pips=self.config.tp_pips,
            sl_pips=self.config.sl_pips,
            max_horizon=self.config.max_horizon,
            price_col="close",
            pip_value=self.config.pip_value,
        )

        y = pd.Series(tb, index=candles.index).reindex(X.index).dropna()

        if self.config.drop_neutral:
            y = y[y != 0]

        return y.astype(float)

    # ------------------------------------------------------------------ #
    # Multi-timeframe enrichment (candle mode)                           #
    # ------------------------------------------------------------------ #

    def _maybe_enrich_multi_tf(
        self,
        candles: pd.DataFrame,
        X: pd.DataFrame,
        y: pd.Series,
        regimes: Optional[pd.DataFrame],
    ) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame]]:
        if not self.config.multi_tf_timeframes:
            return X, y, regimes

        targets = []
        for tf in self.config.multi_tf_timeframes:
            try:
                minutes = int(tf)
                targets.append((tf, minutes))
            except ValueError:
                continue

        if not targets:
            return X, y, regimes

        aggregator = MultiTFAggregatorV4(config=MultiTFConfigV4(targets=targets))
        tf_features = aggregator.aggregate_history(candles)
        tf_features = {
            tf: df for tf, df in tf_features.items()
            if df is not None and not df.empty
        }

        if not tf_features:
            return X, y, regimes

        merged = None
        for tf_name, df in tf_features.items():
            df = df.copy()
            df.index = candles.index[: len(df)]
            df = df.reindex(X.index)
            df = df.add_prefix(f"tf_{tf_name}_")
            merged = df if merged is None else merged.join(df, how="inner")

        if merged is None or merged.empty:
            return X, y, regimes

        X_joined = X.join(merged, how="inner")
        if X_joined.empty:
            return X, y, regimes

        new_index = X_joined.index
        y_aligned = y.reindex(new_index).dropna()
        X_final = X_joined.loc[y_aligned.index]

        regimes_final = None
        if regimes is not None and not regimes.empty:
            regimes_final = regimes.reindex(X_final.index)

        return X_final, y_aligned, regimes_final

    # ------------------------------------------------------------------ #
    # Weights                                                            #
    # ------------------------------------------------------------------ #

    def _apply_weights(
        self,
        y: pd.Series,
        regimes: Optional[pd.DataFrame],
        sample_weight: Optional[pd.Series],
    ) -> np.ndarray:
        n = len(y)
        w = np.ones(n, dtype=float)

        if sample_weight is not None:
            sw = sample_weight.reindex(y.index).fillna(0.0)
            w *= sw.values

        if self.config.sample_weights is not None:
            sw = self.config.sample_weights.reindex(y.index).fillna(0.0)
            w *= sw.values

        if self.config.regime_weights and regimes is not None and not regimes.empty:
            reg = regimes.reindex(y.index)
            if "regime_label" in reg.columns:
                labels = reg["regime_label"].astype(str)
                for regime_label, weight in self.config.regime_weights.items():
                    mask = labels == str(regime_label)
                    w[mask.values] *= float(weight)

        return w

    # ------------------------------------------------------------------ #
    # Model training                                                     #
    # ------------------------------------------------------------------ #

    def _train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: np.ndarray,
    ) -> Any:
        model = self.model_factory()

        if hasattr(model, "set_params") and self.config.hyperparameters:
            model.set_params(**self.config.hyperparameters)

        fit_kwargs: Dict[str, Any] = {}
        if sample_weights is not None:
            fit_kwargs["sample_weight"] = sample_weights

        model.fit(X.values, y.values, **fit_kwargs)
        return model

    # ------------------------------------------------------------------ #
    # Metrics + diagnostics                                              #
    # ------------------------------------------------------------------ #

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
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        regimes: Optional[pd.DataFrame],
        strategies: Optional[Dict[str, pd.DataFrame]],
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        scores = self._predict_scores(model, X)
        pred_df = pd.DataFrame({"prediction": scores}, index=X.index)

        y_true = y.values
        y_pred_class = np.where(scores > 0, 1, 0)

        basic = self._compute_basic_metrics(y_true, y_pred_class)
        extended = self._compute_extended_metrics(y, pred_df, scores, y_pred_class)

        metrics: Dict[str, Any] = {**basic, **extended}

        if self.config.compute_feature_importance:
            metrics["feature_importance"] = self._compute_feature_importance(model, X)

        if self.config.compute_regime_performance:
            metrics["regime_performance"] = self._compute_regime_performance(
                y_true, y_pred_class, regimes, X.index
            )

        if self.config.compute_strategy_performance:
            metrics["strategy_performance"] = self._compute_strategy_performance(
                y_true, y_pred_class, strategies, X.index, X
            )

        return metrics, pred_df

    @staticmethod
    def _compute_basic_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }

    @staticmethod
    def _compute_extended_metrics(
        y: pd.Series,
        pred_df: pd.DataFrame,
        scores: np.ndarray,
        y_pred_class: np.ndarray,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        try:
            if len(np.unique(y.values)) > 1:
                out["auc"] = float(roc_auc_score(y.values, scores))
        except Exception:
            out["auc"] = float("nan")

        try:
            tn, fp, fn, tp = confusion_matrix(
                y.values, y_pred_class, labels=[0, 1]
            ).ravel()
            out.update({
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
            })
        except Exception:
            out.update({"tp": 0, "tn": 0, "fp": 0, "fn": 0})

        return out

    @staticmethod
    def _compute_regime_performance(
        y_true: np.ndarray,
        y_pred_class: np.ndarray,
        regimes: Optional[pd.DataFrame],
        index: pd.Index,
    ) -> Dict[str, Any]:
        if regimes is None or regimes.empty:
            return {}

        reg = regimes.reindex(index)
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

            metrics_r = UnifiedTrainingPipeline._compute_basic_metrics(y_r, y_pred_r)
            performance[str(regime_label)] = metrics_r

        return performance

    @staticmethod
    def _compute_strategy_performance(
        y_true: np.ndarray,
        y_pred_class: np.ndarray,
        strategies: Optional[Dict[str, pd.DataFrame]],
        index: pd.Index,
        X: Optional[pd.DataFrame],
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

    @staticmethod
    def _compute_feature_importance(
        model: Any,
        X: pd.DataFrame,
    ) -> Dict[str, float]:
        importances: Dict[str, float] = {}

        if hasattr(model, "feature_importances_"):
            raw = model.feature_importances_
            if len(raw) == X.shape[1]:
                for col, val in zip(X.columns, raw):
                    importances[col] = float(val)
        elif hasattr(model, "get_booster"):
            try:
                booster = model.get_booster()
                score = booster.get_score(importance_type="gain")
                for k, v in score.items():
                    if k.startswith("f"):
                        idx = int(k[1:])
                        if idx < len(X.columns):
                            importances[X.columns[idx]] = float(v)
            except Exception:
                pass

        total = sum(importances.values())
        if total > 0:
            importances = {k: v / total for k, v in importances.items()}

        return importances