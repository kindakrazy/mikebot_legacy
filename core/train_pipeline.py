# mikebot/core/train_pipeline.py

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from mikebot.core.feature_builder import FeatureBuilder, FeatureConfig
from mikebot.core.model_registry import (
    load_registry,
    save_registry,
    update_entry,
)
from mikebot.core.model_lineage import ModelLineageRegistry
from mikebot.core.experiment_record import ExperimentRecord
from mikebot.strategies.registry import compute_all_filtered
from mikebot.config.strategy_config import load_strategy_toggles
from mikebot.core.triple_barrier import triple_barrier_labeling
from mikebot.core.experience_store import ExperienceStore
from mikebot.core.feature_aggregator import FeatureAggregator


# ======================================================================
# TrainingConfig
# ======================================================================

@dataclass
class TrainingConfig:
    """
    Unified training configuration.

    Supports:
    - experiment hyperparameters
    - sample weights
    - strategy features
    - regime weights
    - warm-start models
    - experiment metadata
    """

    hyperparameters: Dict[str, Any]
    sample_weights: Optional[np.ndarray] = None
    strategy_features: Optional[Dict[str, pd.DataFrame]] = None
    regime_weights: Optional[Dict[str, float]] = None
    warm_start_model: Optional[Path] = None
    experiment_type: str = "baseline"
    notes: Optional[str] = None


# ======================================================================
# Metrics
# ======================================================================

def compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Basic Highstrike metrics."""
    if len(y_true) == 0:
        return {
            "win_rate": 0.0,
            "accuracy": 0.0,
            "directional_hit_rate": 0.0,
        }

    accuracy = (y_true == y_pred).mean()

    mask = y_pred != 0
    win_rate = (y_true[mask] == y_pred[mask]).mean() if mask.sum() > 0 else 0.0

    directional = (np.sign(y_true) == np.sign(y_pred)).mean()

    return {
        "win_rate": float(win_rate),
        "accuracy": float(accuracy),
        "directional_hit_rate": float(directional),
    }


def compute_extended_metrics(y: pd.Series, pred_df: pd.DataFrame) -> Dict[str, float]:
    """Extended metrics from canonical pipeline."""
    p = pred_df["prediction"]
    y_bin = (y > 0).astype(int)
    p_bin = (p > 0).astype(int)

    accuracy = float((y_bin == p_bin).mean())
    win_rate = float((p_bin == 1).mean())
    directional_hit = float((np.sign(y) == np.sign(p)).mean())
    rmse = float(np.sqrt(((y - p) ** 2).mean()))

    return {
        "accuracy_ext": accuracy,
        "win_rate_ext": win_rate,
        "directional_hit_ext": directional_hit,
        "rmse": rmse,
    }


# ======================================================================
# Auto-promotion logic
# ======================================================================

def should_promote(new_metrics: Dict, old_metrics: Optional[Dict]) -> bool:
    if old_metrics is None:
        return True

    return (
        new_metrics.get("win_rate", 0.0) >= old_metrics.get("win_rate", 0.0)
        and new_metrics.get("accuracy", 0.0) >= old_metrics.get("accuracy", 0.0)
    )


# ======================================================================
# Canonical symbol + strict S3 timeframe detection
# ======================================================================

def detect_symbol_and_timeframe_from_normalized(normalized_path: Path) -> Tuple[str, str]:
    """
    Extract canonical symbol and strict S3 timeframe from a normalized CSV.

    - Symbol: strip trailing digits from filename stem (BTCUSD1 → BTCUSD)
    - Timeframe: detect from timestamp spacing
      Strict S3 mode:
        - allow up to 2 irregular intervals
        - base delta must map to canonical TF (M1, M5, M15, M30, H1, H4, D1)
    """
    import re
    from collections import Counter

    # 1. Canonical symbol
    raw_stem = normalized_path.stem.split(".")[0]
    symbol = re.sub(r"\d+$", "", raw_stem).upper()

    # 2. Load timestamps
    df = pd.read_csv(normalized_path)
    if "timestamp" not in df.columns:
        raise ValueError("Normalized file must contain a 'timestamp' column.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")

    if len(df) < 3:
        raise ValueError("Not enough rows to determine timeframe.")

    # 3. Compute deltas (minutes)
    deltas = (df.index[1:] - df.index[:-1]).total_seconds() / 60.0
    if len(deltas) == 0:
        raise ValueError("Not enough timestamps to compute intervals.")

    from collections import Counter
    delta_counts = Counter(deltas)
    base_delta, _ = delta_counts.most_common(1)[0]

    irregular = [d for d in deltas if d != base_delta]
    if len(irregular) > 2:
        raise ValueError(
            f"Strict timeframe detection failed: {len(irregular)} irregular intervals "
            f"(allowed up to 2). Unique deltas: {sorted(set(deltas))}"
        )

    # 4. Map to canonical TF
    delta_to_tf = {
        1: "M1",
        5: "M5",
        15: "M15",
        30: "M30",
        60: "H1",
        240: "H4",
        1440: "D1",
    }

    if base_delta not in delta_to_tf:
        raise ValueError(f"Unknown timeframe: {base_delta} minutes")

    timeframe = delta_to_tf[base_delta]

    print(
        f"[TRAIN] Detected symbol={symbol}, timeframe={timeframe} "
        f"(base_delta={base_delta}, irregular={len(irregular)})"
    )
    return symbol, timeframe


# ======================================================================
# Unified TrainPipeline (symbol-level, multi-timeframe aware)
# ======================================================================

class TrainPipeline:
    """
    Unified training pipeline for Mikebot.

    Public APIs:
    - train(...)          # legacy Highstrike-style APIs
    - train_simple(...)   # legacy Highstrike-style APIs
    - train_multitf(...)  # NEW: MetaTrainer / MULTITF-friendly API

    All legacy logic flows through _train_core().
    The new train_multitf() uses a simpler, symbol-level path.
    """

    def __init__(
        self,
        model_dir: Path,
        registry_path: Path,
        experience_store: Optional[ExperienceStore] = None,
        feature_aggregator: Optional[FeatureAggregator] = None,
        lineage_registry: Optional[ModelLineageRegistry] = None,
    ):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.registry_path = registry_path
        self.experience_store = experience_store
        self.feature_aggregator = feature_aggregator
        self.lineage_registry = lineage_registry

    # ------------------------------------------------------------------
    # CANONICAL API (legacy)
    # ------------------------------------------------------------------

    def train(
        self,
        symbol: str,
        timeframe: str,
        model_type: str,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        regimes: Optional[pd.DataFrame],
        strategies: Dict[str, pd.DataFrame],
        config: TrainingConfig,
    ) -> Dict[str, Any]:

        return self._train_core(
            symbol=symbol,
            timeframe=timeframe,
            model_type=model_type,
            raw_candles=None,
            features=features,
            labels=labels,
            regimes=regimes,
            strategies=strategies,
            config=config,
        )

    # ------------------------------------------------------------------
    # DEVELOPMENT API (legacy)
    # ------------------------------------------------------------------

    def train_simple(self, cfg: TrainingConfig, raw_candles: pd.DataFrame) -> Dict:
        """
        Legacy HighstrikeSignals-style API.
        Delegates to the canonical _train_core implementation.
        """
        return self._train_core(
            symbol=cfg.hyperparameters.get("symbol", "UNKNOWN"),
            timeframe=cfg.hyperparameters.get("timeframe", "UNKNOWN"),
            model_type=cfg.hyperparameters.get("model_type", "rf"),
            raw_candles=raw_candles,
            features=None,
            labels=None,
            regimes=None,
            strategies=None,
            config=cfg,
        )

    # ------------------------------------------------------------------
    # NEW API: MetaTrainer / MULTITF-friendly
    # ------------------------------------------------------------------

    def train_multitf(
        self,
        symbol: str,
        timeframes: List[str],
        model_type: str,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        regimes: Optional[pd.DataFrame],
        strategies: Optional[Dict[str, pd.DataFrame]],
        config: TrainingConfig,
    ) -> Dict[str, Any]:
        """
        Symbol-level MULTITF training path for MetaTrainer.

        Assumptions:
        - `features` is already a prepared feature matrix (no FeatureBuilder / triple-barrier).
        - `labels` provides the target; we use:
            - labels["target"] if present, else
            - the first column.
        - Registry auto-promotion is intentionally skipped here; MetaTrainer controls promotion.

        Returns:
            {
                "predictions": DataFrame,
                "errors": DataFrame,
                "metrics": Dict[str, Any],
                "regime_performance": Dict[str, Any],
                "strategy_performance": Dict[str, Any],
                "model_path": Path,
                "metrics_path": Path,
            }
        """
        if features is None or labels is None:
            raise ValueError("train_multitf requires `features` and `labels` to be provided.")

        # Base timeframe for logging; multi-TF list is used only as metadata/hints.
        base_timeframe = str(timeframes[0]) if timeframes else "UNKNOWN"

        # Ensure multi-TF timeframes are present in hyperparameters as numeric hints.
        tf_numeric = [self._timeframe_to_numeric_str(str(tf)) for tf in timeframes]
        config.hyperparameters.setdefault("multi_tf_timeframes", tf_numeric)

        print(
            f"[TRAIN-MULTITF] Training symbol-level MULTITF model for {symbol} "
            f"over timeframes={timeframes} using base timeframe={base_timeframe}"
        )

        # 1. Build X, y directly from provided frames
        X = features.copy()

        if "target" in labels.columns:
            y = labels["target"].reindex(X.index)
        else:
            y = labels.iloc[:, 0].reindex(X.index)

        # Drop NaNs in y and align X
        mask = y.notna()
        X = X.loc[mask]
        y = y.loc[mask]

        print(f"[TRAIN-MULTITF] Feature matrix: {X.shape[0]} samples, {X.shape[1]} features")

        # 2. (Optional) Multi-TF enrichment via ExperienceStore + FeatureAggregator
        X, y, regimes = self._maybe_enrich_with_multi_tf(
            symbol=symbol,
            timeframe=base_timeframe,
            X=X,
            y=y,
            regimes=regimes,
            config=config,
        )

        # 3. Sample/regime weights
        print("[TRAIN-MULTITF] Applying sample/regime weights (if configured)...")
        sample_weights = self._apply_weights(y, regimes, config)
        print(
            f"[TRAIN-MULTITF] Weights applied. Non-zero weights: "
            f"{(sample_weights != 0).sum()} / {len(sample_weights)}"
        )

        # 4. Train model
        print(
            f"[TRAIN-MULTITF] Training SYMBOL-LEVEL MULTITF model for {symbol}: "
            f"{model_type.upper()} on {len(X)} samples, {X.shape[1]} features"
        )
        model = self._train_model(model_type, X, y, config, sample_weights)
        print("[TRAIN-MULTITF] Model training complete.")

        # 5. Predictions + metrics
        print("[TRAIN-MULTITF] Computing predictions and metrics...")
        preds = self._predict(model, X)
        pred_df = pd.DataFrame({"prediction": preds}, index=X.index)

        y_pred_class = np.where(preds > 0, 1, 0)

        basic = compute_basic_metrics(y.values, y_pred_class)
        extended = compute_extended_metrics(y, pred_df)

        feature_importance = self._compute_feature_importance(model, X)

        regime_performance = self._compute_regime_performance(
            y_true=y.values,
            y_pred_class=y_pred_class,
            regimes=regimes,
            index=X.index,
        )

        strategy_performance = self._compute_strategy_performance(
            y_true=y.values,
            y_pred_class=y_pred_class,
            strategies=strategies or {},
            index=X.index,
            X=X,
        )

        metrics = {
            **basic,
            **extended,
            "feature_importance": feature_importance,
            "regime_performance": regime_performance,
            "strategy_performance": strategy_performance,
        }

        print(f"[TRAIN-MULTITF] Basic metrics: {basic}")
        print(f"[TRAIN-MULTITF] Extended metrics: {extended}")
        if feature_importance:
            print(
                f"[TRAIN-MULTITF] Top feature importance (sample): "
                f"{dict(list(feature_importance.items())[:5])}"
            )

        # 6. Save artifacts (symbol-level)
        version_tag = self._version_tag(symbol)
        print(f"[TRAIN-MULTITF] Version tag: {version_tag}")

        model_path = self._save_model(model, symbol, model_type, version_tag)
        metrics_path = self._save_metrics(metrics, symbol, model_type, version_tag)

        print(f"[TRAIN-MULTITF] Saved model to:   {model_path}")
        print(f"[TRAIN-MULTITF] Saved metrics to: {metrics_path}")

        # IMPORTANT: NO REGISTRY UPDATE HERE.
        # MetaTrainer controls promotion and registry updates for MULTITF models.

        # 7. Lineage emission (v4, symbol-level MULTITF)
        if self.lineage_registry is not None:
            try:
                parent_version_id: Optional[str] = None
                latest = self.lineage_registry.get_latest(symbol, base_timeframe, model_type)
                if latest is not None:
                    parent_version_id = latest[0]

                record = ExperimentRecord.from_pipeline_outputs(
                    symbol=symbol,
                    timeframe=base_timeframe,
                    model_type=model_type,
                    version_id="",  # let ModelLineageRegistry assign sequential vN
                    parent_version_id=parent_version_id,
                    metrics=metrics,
                    experiment_type=config.experiment_type,
                    notes=config.notes or "",
                    model_path=model_path,
                    metrics_path=metrics_path,
                    model_tag=version_tag,
                )

                self.lineage_registry.record_experiment(record)
                print("[TRAIN-MULTITF] Lineage record emitted.")
            except Exception as e:
                print(f"[TRAIN-MULTITF] WARNING: Failed to record lineage: {e}")

        # Build errors DataFrame
        errors_df = (y - pred_df["prediction"]).to_frame(name="error")

        return {
            "predictions": pred_df,
            "errors": errors_df,
            "metrics": metrics,
            "regime_performance": regime_performance,
            "strategy_performance": strategy_performance,
            "model_path": model_path,
            "metrics_path": metrics_path,
        }

    # ------------------------------------------------------------------
    # CORE LEGACY TRAINING FLOW
    # ------------------------------------------------------------------

    def _train_core(
        self,
        symbol: str,
        timeframe: str,
        model_type: str,
        raw_candles: Optional[pd.DataFrame],
        features: Optional[pd.DataFrame],
        labels: Optional[pd.DataFrame],
        regimes: Optional[pd.DataFrame],
        strategies: Optional[Dict[str, pd.DataFrame]],
        config: TrainingConfig,
    ) -> Dict[str, Any]:
        """
        Centralized training flow used by both legacy APIs.
        """

        # --------------------------------------------------------------
        # 1. Build single-timeframe features (existing behavior)
        # --------------------------------------------------------------
        toggles = load_strategy_toggles()
        strategy_feature_summary: Dict[str, Any] = {}
        feature_origin: Dict[str, Dict[str, str]] = {}
        strategy_feature_diagnostics: Dict[str, Any] = {}
        strategies_local: Dict[str, pd.DataFrame] = strategies or {}

        if raw_candles is not None:
            print(f"[TRAIN] Building features from raw candles: {len(raw_candles)} rows")

            try:
                strategy_signals = compute_all_filtered(raw_candles, toggles)
            except Exception:
                strategy_signals = {}
            strategies_local = strategy_signals

            fb = FeatureBuilder(FeatureConfig())
            feats = fb.build_features(
                candles=raw_candles,
                strategies=strategy_signals,
                regimes=regimes,
                experiment_config=config.hyperparameters,
            )

            try:
                strategy_feature_summary = fb.get_last_strategy_feature_summary()
            except Exception:
                strategy_feature_summary = getattr(fb, "_last_strategy_feature_summary", {})

            try:
                feature_origin = fb.get_feature_origin_map()
            except Exception:
                feature_origin = getattr(fb, "_last_feature_origin_map", {})

            try:
                strategy_feature_diagnostics = fb.get_last_strategy_feature_diagnostics()
            except Exception:
                strategy_feature_diagnostics = getattr(
                    fb, "_last_strategy_feature_diagnostics", {}
                )

            print(f"[TRAIN] Strategy feature summary keys: {list(strategy_feature_summary.keys())}")

            print(
                f"[TRAIN] Feature matrix built from raw candles: "
                f"{feats.shape[0]} samples, {feats.shape[1]} columns"
            )

            # Triple‑Barrier target (raw-candles path only)
            tp_pips = config.hyperparameters.get("tp_pips", 20)
            sl_pips = config.hyperparameters.get("sl_pips", 10)
            max_horizon = config.hyperparameters.get("max_horizon", 12)
            pip_value = config.hyperparameters.get("pip_value", 0.0001)

            feats["target"] = triple_barrier_labeling(
                df=feats,
                tp_pips=tp_pips,
                sl_pips=sl_pips,
                max_horizon=max_horizon,
                price_col="close",
                pip_value=pip_value,
            )

            print("[TRAIN] Triple-barrier target distribution:", feats["target"].value_counts().to_dict())

            feats = feats.dropna()
            if config.hyperparameters.get("drop_neutral", False):
                feats = feats[feats["target"] != 0]

            X = feats.drop(columns=["target"])
            y = feats["target"]

        else:
            # Precomputed feature path
            print(f"[TRAIN] Using precomputed features/labels")

            strategies_local = strategies or {}

            fb = FeatureBuilder(FeatureConfig())
            X = fb.build_features(
                candles=features,
                strategies=strategies_local,
                regimes=regimes,
                experiment_config=config.hyperparameters,
            )

            try:
                strategy_feature_summary = fb.get_last_strategy_feature_summary()
            except Exception:
                strategy_feature_summary = getattr(fb, "_last_strategy_feature_summary", {})

            try:
                feature_origin = fb.get_feature_origin_map()
            except Exception:
                feature_origin = getattr(fb, "_last_feature_origin_map", {})

            try:
                strategy_feature_diagnostics = fb.get_last_strategy_feature_diagnostics()
            except Exception:
                strategy_feature_diagnostics = getattr(
                    fb, "_last_strategy_feature_diagnostics", {}
                )

            print(f"[TRAIN] Strategy feature summary keys: {list(strategy_feature_summary.keys())}")

            if labels is None:
                raise ValueError("Precomputed feature path requires `labels` to be provided.")

            if hasattr(labels, "index") and labels.index.equals(X.index):
                if "target" in labels.columns:
                    y = labels.loc[X.index, "target"]
                else:
                    y = labels.loc[X.index].squeeze().astype(float)
            else:
                import warnings
                warnings.warn(
                    "labels.index does not match X.index — falling back to positional label extraction",
                    UserWarning,
                )
                y = labels.iloc[:, 0].astype(float)

            print(f"[TRAIN] Feature matrix: {X.shape[0]} samples, {X.shape[1]} features")

        # --------------------------------------------------------------
        # 1b. Multi-timeframe enrichment (symbol-level)
        # --------------------------------------------------------------
        X, y, regimes = self._maybe_enrich_with_multi_tf(
            symbol=symbol,
            timeframe=timeframe,
            X=X,
            y=y,
            regimes=regimes,
            config=config,
        )

        # --------------------------------------------------------------
        # 2. Sample weights
        # --------------------------------------------------------------
        print("[TRAIN] Applying sample/regime weights (if configured)...")
        sample_weights = self._apply_weights(y, regimes, config)
        print(f"[TRAIN] Weights applied. Non-zero weights: {(sample_weights != 0).sum()} / {len(sample_weights)}")

        # --------------------------------------------------------------
        # 3. Train model (symbol-level)
        # --------------------------------------------------------------
        print(f"[TRAIN] Training SYMBOL-LEVEL model for {symbol}: {model_type.upper()} on {len(X)} samples, {X.shape[1]} features")
        model = self._train_model(model_type, X, y, config, sample_weights)
        print("[TRAIN] Model training complete.")

        # --------------------------------------------------------------
        # 4. Predictions + metrics
        # --------------------------------------------------------------
        print("[TRAIN] Computing predictions and metrics...")
        preds = self._predict(model, X)
        pred_df = pd.DataFrame({"prediction": preds}, index=X.index)

        y_pred_class = np.where(preds > 0, 1, 0)

        basic = compute_basic_metrics(y.values, y_pred_class)
        extended = compute_extended_metrics(y, pred_df)

        feature_importance = self._compute_feature_importance(model, X)

        regime_performance = self._compute_regime_performance(
            y_true=y.values,
            y_pred_class=y_pred_class,
            regimes=regimes,
            index=X.index,
        )

        strategy_performance = self._compute_strategy_performance(
            y_true=y.values,
            y_pred_class=y_pred_class,
            strategies=strategies_local,
            index=X.index,
            X=X,
        )

        metrics = {
            **basic,
            **extended,
            "feature_importance": feature_importance,
            "regime_performance": regime_performance,
            "strategy_performance": strategy_performance,
        }

        print(f"[TRAIN] Basic metrics: {basic}")
        print(f"[TRAIN] Extended metrics: {extended}")
        if feature_importance:
            print(f"[TRAIN] Top feature importance (sample): {dict(list(feature_importance.items())[:5])}")

        # --------------------------------------------------------------
        # 5. Save artifacts (symbol-level)
        # --------------------------------------------------------------
        version_tag = self._version_tag(symbol)
        print(f"[TRAIN] Version tag: {version_tag}")

        model_path = self._save_model(model, symbol, model_type, version_tag)
        metrics_path = self._save_metrics(metrics, symbol, model_type, version_tag)

        print(f"[TRAIN] Saved model to:   {model_path}")
        print(f"[TRAIN] Saved metrics to: {metrics_path}")

        # --------------------------------------------------------------
        # 6. Registry update + auto-promotion (symbol::MULTITF)
        # --------------------------------------------------------------
        print("[TRAIN] Updating model registry and evaluating promotion...")
        registry_entry = self._update_registry(symbol, model_type, model_path, metrics)

        promoted_flag = registry_entry.get("promoted", False)
        print(f"[TRAIN] Registry updated. Promoted: {promoted_flag}")

        # NOTE: v4 lineage integration for legacy path can be added later
        # once you decide how much of the old flow should be in the tree.

        return {
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
            "metrics": metrics,
            "registry_entry": registry_entry,
        }

    # ------------------------------------------------------------------
    # MULTI-TIMEFRAME ENRICHMENT
    # ------------------------------------------------------------------

    def _maybe_enrich_with_multi_tf(
        self,
        symbol: str,
        timeframe: str,
        X: pd.DataFrame,
        y: pd.Series,
        regimes: Optional[pd.DataFrame],
        config: TrainingConfig,
    ) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame]]:
        """
        Load multi-TF features from ExperienceStore and join them onto X,
        then reindex y and regimes accordingly.

        - Derives a base numeric TF from `timeframe` (e.g., "M5" -> "5") but
          this is only a hint; the model itself is symbol-level.
        - Timeframe list comes from:
            config.hyperparameters["multi_tf_timeframes"]
          or defaults to [base_tf_str].
        """
        if self.experience_store is None or self.feature_aggregator is None:
            return X, y, regimes

        base_tf_str = self._timeframe_to_numeric_str(timeframe)
        tf_list: List[str] = config.hyperparameters.get("multi_tf_timeframes") or [base_tf_str]
        tf_list = [str(tf) for tf in tf_list]

        print(f"[TRAIN] Multi-TF enabled (symbol-level). Loading timeframes for {symbol}: {tf_list}")

        tf_features = self.experience_store.load_multi_tf_features(symbol, tf_list)
        tf_features = {tf: df for tf, df in tf_features.items() if df is not None and not df.empty}

        if not tf_features:
            print("[TRAIN] Multi-TF: no usable timeframe features found in ExperienceStore; using single-TF X only.")
            return X, y, regimes

        merged_multi_tf = self.feature_aggregator.aggregate_multi_tf(tf_features)

        print(
            f"[TRAIN] Multi-TF merged feature matrix: "
            f"{merged_multi_tf.shape[0]} samples, {merged_multi_tf.shape[1]} columns"
        )

        X_joined = X.join(merged_multi_tf, how="inner")

        if X_joined.empty:
            print("[TRAIN] Multi-TF: inner join produced empty matrix; reverting to single-TF X only.")
            return X, y, regimes

        new_index = X_joined.index

        y_aligned = y.reindex(new_index)
        y_aligned = y_aligned.dropna()
        X_final = X_joined.loc[y_aligned.index]

        regimes_final = None
        if regimes is not None and not regimes.empty:
            regimes_final = regimes.reindex(X_final.index)

        print(
            f"[TRAIN] Multi-TF enrichment complete (symbol-level): "
            f"{X_final.shape[0]} samples, {X_final.shape[1]} features after join"
        )

        return X_final, y_aligned, regimes_final

    @staticmethod
    def _timeframe_to_numeric_str(timeframe: str) -> str:
        """
        Convert timeframe labels like 'M1', 'M5', 'M15', 'H1', 'H4', 'D1'
        into simple numeric minute strings ('1', '5', '15', '60', '240', '1440').

        If parsing fails, returns `timeframe` unchanged.
        """
        tf = timeframe.upper()
        mapping = {
            "M1": "1",
            "M5": "5",
            "M15": "15",
            "M30": "30",
            "H1": "60",
            "H4": "240",
            "D1": "1440",
        }
        return mapping.get(tf, timeframe)

    # ------------------------------------------------------------------
    # WEIGHTS / MODEL / PREDICT
    # ------------------------------------------------------------------

    def _apply_weights(
        self,
        y: pd.Series,
        regimes: Optional[pd.DataFrame],
        config: TrainingConfig,
    ) -> np.ndarray:
        """
        Combine:
        - explicit sample_weights from config
        - optional regime_weights from config.regime_weights applied using regimes DataFrame

        Returns a 1D numpy array of length len(y).
        """
        n = len(y)
        w = np.ones(n, dtype=float)

        if config.sample_weights is not None:
            if len(config.sample_weights) != n:
                raise ValueError(
                    f"TrainPipeline._apply_weights: sample_weights length {len(config.sample_weights)} "
                    f"does not match y length {n}"
                )
            w *= config.sample_weights

        if config.regime_weights and regimes is not None and not regimes.empty:
            reg = regimes.copy()
            reg = reg.reindex(y.index)

            if "regime_label" in reg.columns:
                labels = reg["regime_label"].astype(str)
                for regime_label, weight in config.regime_weights.items():
                    mask = labels == str(regime_label)
                    w[mask.values] *= float(weight)

        return w

    def _train_model(
        self,
        model_type: str,
        X: pd.DataFrame,
        y: pd.Series,
        config: TrainingConfig,
        sample_weights: np.ndarray,
    ) -> Any:
        """
        Train a model (RF or XGB) using hyperparameters in config.hyperparameters.
        """
        params = dict(config.hyperparameters)

        for k in [
            "symbol",
            "timeframe",
            "model_type",
            "tp_pips",
            "sl_pips",
            "max_horizon",
            "pip_value",
            "drop_neutral",
            "multi_tf_timeframes",
        ]:
            params.pop(k, None)

        model_type = model_type.lower()

        if model_type in ("rf", "random_forest"):
            n_estimators = int(params.pop("n_estimators", 200))
            max_depth = params.pop("max_depth", None)
            if max_depth is not None:
                max_depth = int(max_depth)

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                n_jobs=params.pop("n_jobs", -1),
                random_state=params.pop("random_state", 42),
                **params,
            )
            model.fit(X.values, y.values, sample_weight=sample_weights)
            return model

        elif model_type in ("xgb", "xgboost"):
            model = XGBClassifier(
                n_estimators=int(params.pop("n_estimators", 400)),
                max_depth=int(params.pop("max_depth", 4)),
                learning_rate=float(params.pop("learning_rate", 0.05)),
                subsample=float(params.pop("subsample", 0.9)),
                colsample_bytree=float(params.pop("colsample_bytree", 0.9)),
                tree_method=params.pop("tree_method", "hist"),
                random_state=params.pop("random_state", 42),
                n_jobs=params.pop("n_jobs", -1),
                **params,
            )
            model.fit(X.values, y.values, sample_weight=sample_weights)
            return model

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    @staticmethod
    def _predict(model: Any, X: pd.DataFrame) -> np.ndarray:
        """
        Produce continuous prediction scores.
        """
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X.values)
            if proba.shape[1] == 2:
                return proba[:, 1] - proba[:, 0]
            else:
                return np.argmax(proba, axis=1).astype(float)
        elif hasattr(model, "decision_function"):
            return model.decision_function(X.values).astype(float)
        else:
            return model.predict(X.values).astype(float)

    # ------------------------------------------------------------------
    # SAVE ARTIFACTS (symbol-level)
    # ------------------------------------------------------------------

    def _version_tag(self, symbol: str) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{symbol}_{ts}"

    def _save_model(self, model, symbol: str, model_type: str, version_tag: str) -> Path:
        out = self.model_dir / f"{symbol}_MULTITF_{model_type}_{version_tag}.bin"
        with out.open("wb") as f:
            pickle.dump(model, f)
        return out

    def _save_metrics(self, metrics, symbol: str, model_type: str, version_tag: str) -> Path:
        out = self.model_dir / f"{symbol}_MULTITF_{model_type}_{version_tag}_metrics.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        return out

    # ------------------------------------------------------------------
    # REGISTRY UPDATE (symbol::MULTITF) – legacy path only
    # ------------------------------------------------------------------

    def _update_registry(self, symbol: str, model_type: str, model_path: Path, metrics: Dict[str, Any]) -> Dict[str, Any]:
        registry = load_registry(self.registry_path)

        key = f"{symbol}::MULTITF::{model_type}"
        old_entry = registry["models"].get(key)
        old_metrics = old_entry.get("metrics") if old_entry else None

        promote = should_promote(metrics, old_metrics)

        entry = update_entry(
            registry=registry,
            key=key,
            model_path=str(model_path),
            metrics=metrics,
            promoted=promote,
        )

        save_registry(self.registry_path, registry)
        return entry

    # ------------------------------------------------------------------
    # REGIME PERFORMANCE
    # ------------------------------------------------------------------

    def _compute_regime_performance(
        self,
        y_true: np.ndarray,
        y_pred_class: np.ndarray,
        regimes: Optional[pd.DataFrame],
        index: pd.Index,
    ) -> Dict[str, Any]:
        if regimes is None or regimes.empty:
            return {}

        reg = regimes.copy()
        reg = reg.reindex(index)

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
            metrics_r = compute_basic_metrics(y_r, y_pred_r)
            performance[str(regime_label)] = metrics_r

        return performance

    # ------------------------------------------------------------------
    # STRATEGY PERFORMANCE
    # ------------------------------------------------------------------

    def _compute_strategy_performance(
        self,
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

    # ------------------------------------------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------------------------------------------

    def _compute_feature_importance(
        self,
        model: Any,
        X: pd.DataFrame,
    ) -> Dict[str, float]:
        importances: Dict[str, float] = {}

        if hasattr(model, "feature_importances_"):
            raw = model.feature_importances_
            if len(raw) == X.shape[1]:
                for col, val in zip(X.columns, raw):
                    importances[str(col)] = float(val)

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