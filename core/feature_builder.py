# mikebot/core/feature_builder.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Any, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """
    High-level configuration for feature building.

    This is the distilled, canonical view of what used to live across:
    - feeds/atr_feed.py
    - feeds/market_features_feed.py
    - feeds/sentiment_feed.py
    - FeatureForge.json (+ diagnostics copy)
    - volatility_clustering, drift_metrics, sentiment_overlay, multi_timeframe,
      sequencing, trade_sequencer
    """
    atr_window: int = 14
    rsi_window: int = 14
    fast_ma: int = 10
    slow_ma: int = 50
    volume_window: int = 20
    sentiment_window: int = 10
    regime_window: int = 100
    drift_window: int = 200
    multi_tf_lookback: int = 3
    sequencing_window: int = 5
    min_history: int = 250  # minimum rows before features are considered reliable


class FeatureBuilder:
    """
    Unified feature builder.

    Responsibilities (pulled from the old world):
    - Turn raw candles into MarketFeatures-style rows
    - Apply ATR / OHLC / volume / sentiment / regime / drift / multi-TF /
      sequencing features
    - Maintain a feature catalog that can be used for explainability

    Extended with experiment-era capabilities:
    - Strategy features (per-strategy signal columns)
    - Regime features (one-hot regimes + confidence)
    - Experiment-driven feature variants (drop/lag/alt-vol windows)
    - Raw math technical indicators
    - Deterministic validation of final feature matrices
    """

    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        self.config = config or FeatureConfig()
        self._feature_catalog: Dict[str, str] = self._build_feature_catalog()

        # Last-run artifacts for orchestration / lineage
        self._last_strategy_feature_summary: Dict[str, Any] = {}
        self._last_feature_origin_map: Dict[str, Dict[str, str]] = {}

        # Last-run artifacts for orchestration / lineage
        self._last_strategy_feature_summary: Dict[str, Any] = {}
        self._last_feature_origin_map: Dict[str, Dict[str, str]] = {}
        # Per-strategy diagnostics recorded for lineage / debugging
        self._last_strategy_feature_diagnostics: Dict[str, Any] = {}



    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def build_features(
        self,
        candles: pd.DataFrame,
        sentiment: Optional[pd.DataFrame] = None,
        higher_tf_candles: Optional[Dict[str, pd.DataFrame]] = None,
        strategies: Optional[Dict[str, pd.DataFrame]] = None,
        regimes: Optional[pd.DataFrame] = None,
        experiment_config: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Main entry point.

        Parameters
        ----------
        candles:
            DataFrame with at least: ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
            Index may be datetime or integer; 'timestamp' will be normalized.
        sentiment:
            Optional DataFrame with ['timestamp', 'sentiment_score', ...].
        higher_tf_candles:
            Optional dict of timeframe -> candles DataFrame for multi-timeframe confirmation.
        strategies:
            Optional mapping of strategy_name -> DataFrame with at least a 'signal'
            column, aligned by timestamp (will be reindexed to candles).
        regimes:
            Optional DataFrame with at least 'regime_label' and optionally
            'regime_confidence', indexed by timestamp (will be reindexed).
        experiment_config:
            Optional dict of experiment-driven feature instructions, e.g.:
                {
                  "drop_features": [...],
                  "add_lag_features": [1, 2],
                  "add_volatility_window": 20,
                  "enable_raw_math": true,
                }

        Returns
        -------
        DataFrame with engineered features, aligned to the candle index.
        """
        df = candles.copy()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").set_index("timestamp")

        # Basic sanity
        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required candle columns: {missing}")

        logger.debug("Starting feature build on %d rows", len(df))

        # Core technical features

        df = self._add_ohlc_features(df)
        df["ret_1"] = df["close"].pct_change()
        df = self._add_atr(df)
        df = self._add_moving_averages(df)
        df = self._add_rsi(df)
        df = self._add_volume_features(df)

        # Higher-order features
        df = self._add_volatility_clustering(df)
        df = self._add_drift_metrics(df)

        # Sentiment overlay
        if sentiment is not None:
            df = self._add_sentiment_features(df, sentiment)

        # Multi-timeframe confirmation
        if higher_tf_candles:
            df = self._add_multi_timeframe_features(df, higher_tf_candles)

        # Sequencing / trade-sequencer style features
        df = self._add_sequencing_features(df)

        # ---------------------------------------------------------------------
        # Experiment-era enrichments (all optional, do not affect default path)
        # ---------------------------------------------------------------------

        # Strategy features
        if strategies:
            df = self._add_strategy_features(df, strategies)

        # Regime features
        if regimes is not None:
            df = self._add_regime_features(df, regimes)

        # Raw math indicator engine (guarded by experiment_config flag)
        if experiment_config and experiment_config.get("enable_raw_math"):
            raw_math = self._build_raw_math_features(candles)
            # Align by index (timestamps); left join onto df
            raw_math = raw_math.reindex(df.index)
            df = df.join(raw_math, how="left")

        # Experiment-driven adjustments (drop/lag/alt-vol)
        if experiment_config:
            df = self._apply_experiment_feature_config(df, experiment_config)

        # Validation: ensure no NaNs/infs and no empty feature set
        self._validate(df)

        # Drop rows that don't have enough history
        df = df.iloc[self.config.min_history :].copy()

        logger.debug("Completed feature build; resulting rows: %d", len(df))
        return df

    def feature_catalog(self) -> Dict[str, str]:
        """
        Returns a human-readable catalog of features, inspired by modules/ui/explain_panel.FEATURE_CATALOG
        and the knowledge graph relationships.
        """
        return dict(self._feature_catalog)

    def build(self, candles: pd.DataFrame) -> pd.DataFrame:
        """
        Compatibility wrapper expected by TrainPipeline.
        Delegates to the canonical build_features() method.
        """
        return self.build_features(candles)

    # -------------------------------------------------------------------------
    # Snapshot API for live orchestrator
    # -------------------------------------------------------------------------

    def build_for_snapshot(
        self,
        candles_by_symbol: Dict[str, pd.DataFrame],
        sentiment_by_symbol: Optional[Dict[str, pd.DataFrame]] = None,
        higher_tf_by_symbol: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
        strategies_by_symbol: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
        regimes_by_symbol: Optional[Dict[str, pd.DataFrame]] = None,
        experiment_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, object]]:
        """
        Orchestrator-facing API.

        Input:
            candles_by_symbol: {symbol: candles_df}
            sentiment_by_symbol: optional {symbol: sentiment_df}
            higher_tf_by_symbol: optional {symbol: {tf_name: candles_df}}
            strategies_by_symbol: optional {symbol: {strategy_name: df}}
            regimes_by_symbol: optional {symbol: regimes_df}
            experiment_config: optional experiment configuration applied per symbol

        Output:
            {
                symbol: {
                    "features": features_df
                },
                ...
            }
        """
        out: Dict[str, Dict[str, object]] = {}

        for symbol, candles in (candles_by_symbol or {}).items():
            sentiment = None
            if sentiment_by_symbol is not None:
                sentiment = sentiment_by_symbol.get(symbol)

            higher_tf = None
            if higher_tf_by_symbol is not None:
                higher_tf = higher_tf_by_symbol.get(symbol)

            strategies = None
            if strategies_by_symbol is not None:
                strategies = strategies_by_symbol.get(symbol)

            regimes = None
            if regimes_by_symbol is not None:
                regimes = regimes_by_symbol.get(symbol)

            feats = self.build_features(
                candles=candles,
                sentiment=sentiment,
                higher_tf_candles=higher_tf,
                strategies=strategies,
                regimes=regimes,
                experiment_config=experiment_config,
            )

            out[symbol] = {
                "features": feats,
            }

        return out

    # -------------------------------------------------------------------------
    # Core technical features
    # -------------------------------------------------------------------------

    def _add_ohlc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Candle body and wicks
        df["body"] = df["close"] - df["open"]
        df["range"] = df["high"] - df["low"]
        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

        # Normalized body and range
        df["body_pct"] = df["body"] / df["open"].replace(0, np.nan)
        df["range_pct"] = df["range"] / df["open"].replace(0, np.nan)

        return df

    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Approximate ATR implementation, mirroring feeds/atr_feed.py behavior.
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        df["atr"] = tr.rolling(self.config.atr_window, min_periods=1).mean()
        df["atr_pct"] = df["atr"] / df["close"].replace(0, np.nan)

        return df

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        fast = self.config.fast_ma
        slow = self.config.slow_ma

        df[f"ma_{fast}"] = df["close"].rolling(fast, min_periods=1).mean()
        df[f"ma_{slow}"] = df["close"].rolling(slow, min_periods=1).mean()
        df["ma_spread"] = df[f"ma_{fast}"] - df[f"ma_{slow}"]
        df["ma_spread_pct"] = df["ma_spread"] / df[f"ma_{slow}"].replace(0, np.nan)

        return df

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        window = self.config.rsi_window
        delta = df["close"].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window, min_periods=window).mean()
        avg_loss = loss.rolling(window, min_periods=window).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        df["rsi"] = rsi.fillna(50.0)
        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        w = self.config.volume_window
        df["volume_ma"] = df["volume"].rolling(w, min_periods=1).mean()
        df["volume_zscore"] = (df["volume"] - df["volume_ma"]) / (
            df["volume"].rolling(w, min_periods=1).std().replace(0, np.nan)
        )
        return df

    # -------------------------------------------------------------------------
    # Higher-order features: volatility clustering, drift, regimes
    # -------------------------------------------------------------------------

    def _add_volatility_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rough analog of modules/volatility_clustering.py:
        - Use rolling std of returns as a volatility proxy
        - Bucket into regimes
        """
        returns = df["close"].pct_change()
        vol = returns.rolling(self.config.regime_window, min_periods=10).std()
        df["volatility"] = vol

        # Simple quantile-based regimes: 0=low,1=medium,2=high
        q1 = vol.quantile(0.33)
        q2 = vol.quantile(0.66)

        def _regime(v: float) -> int:
            if np.isnan(v):
                return -1
            if v < q1:
                return 0
            if v < q2:
                return 1
            return 2

        df["vol_regime"] = vol.apply(_regime)
        return df

    def _add_drift_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inspired by modules/drift_metrics.py:
        - Rolling mean and std of returns
        - Simple drift score as mean/std
        """
        returns = df["close"].pct_change()
        w = self.config.drift_window

        rolling_mean = returns.rolling(w, min_periods=10).mean()
        rolling_std = returns.rolling(w, min_periods=10).std()

        df["drift_mean"] = rolling_mean
        df["drift_std"] = rolling_std
        df["drift_score"] = rolling_mean / rolling_std.replace(0, np.nan)

        return df

    # -------------------------------------------------------------------------
    # Sentiment overlay
    # -------------------------------------------------------------------------

    def _add_sentiment_features(
        self,
        df: pd.DataFrame,
        sentiment: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge sentiment feed into candle index and build rolling sentiment features.

        Mirrors the intent of feeds/sentiment_feed.py + modules/sentiment_overlay.py.
        """
        s = sentiment.copy()
        if "timestamp" in s.columns:
            s["timestamp"] = pd.to_datetime(s["timestamp"], utc=True)
            s = s.sort_values("timestamp").set_index("timestamp")

        if "sentiment_score" not in s.columns:
            raise ValueError("sentiment DataFrame must contain 'sentiment_score' column")

        # Align sentiment to candle timestamps (forward-fill)
        s = s[["sentiment_score"]].reindex(df.index, method="ffill")

        df["sentiment_score"] = s["sentiment_score"]
        w = self.config.sentiment_window
        df["sentiment_ma"] = df["sentiment_score"].rolling(w, min_periods=1).mean()
        df["sentiment_zscore"] = (df["sentiment_score"] - df["sentiment_ma"]) / (
            df["sentiment_score"].rolling(w, min_periods=1).std().replace(0, np.nan)
        )

        # Overlay: sentiment-adjusted drift
        df["drift_sentiment_adjusted"] = df["drift_score"] * df["sentiment_ma"].fillna(0.0)

        return df

    # -------------------------------------------------------------------------
    # Multi-timeframe confirmation
    # -------------------------------------------------------------------------

    def _add_multi_timeframe_features(
        self,
        df: pd.DataFrame,
        higher_tf_candles: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Approximate modules/multi_timeframe.confirm_signal behavior:
        - For each higher timeframe, compute a simple direction signal
        - Attach as confirmation features
        """
        for tf_name, tf_df in higher_tf_candles.items():
            tf = tf_df.copy()
            if "timestamp" in tf.columns:
                tf["timestamp"] = pd.to_datetime(tf["timestamp"], utc=True)
                tf = tf.sort_values("timestamp").set_index("timestamp")

            if not {"open", "close"}.issubset(tf.columns):
                logger.warning("Higher TF %s missing open/close; skipping", tf_name)
                continue

            tf[f"{tf_name}_direction"] = np.sign(tf["close"] - tf["open"])
            aligned = tf[[f"{tf_name}_direction"]].reindex(df.index, method="ffill")
            df[f"mtf_{tf_name}_direction"] = aligned[f"{tf_name}_direction"]

        # Simple aggregate confirmation score
        mtf_cols = [c for c in df.columns if c.startswith("mtf_") and c.endswith("_direction")]
        if mtf_cols:
            df["mtf_confirmation_score"] = df[mtf_cols].mean(axis=1)

        return df

    # -------------------------------------------------------------------------
    # Sequencing / trade-sequencer style features
    # -------------------------------------------------------------------------

    def _add_sequencing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inspired by modules/sequencing.py and modules/trade_sequencer.py:
        - Encode recent direction sequence as simple pattern features.
        """
        w = self.config.sequencing_window
        direction = np.sign(df["close"].diff()).fillna(0.0)
        df["direction"] = direction

        # Rolling sum of direction (trend strength)
        df["dir_sum_w"] = direction.rolling(w, min_periods=1).sum()

        # Count of up/down candles in window
        df["dir_up_count"] = (direction > 0).rolling(w, min_periods=1).sum()
        df["dir_down_count"] = (direction < 0).rolling(w, min_periods=1).sum()

        # Simple "momentum burst" flag
        df["momentum_burst"] = (df["dir_sum_w"].abs() >= (w - 1)).astype(int)

        return df

    def _add_strategy_features(
            self,
            features: pd.DataFrame,
            strategies: Dict[str, pd.DataFrame],
        ) -> pd.DataFrame:
            """
            Add full strategy feature sets and produce a summary for lineage.

            Behavior changes:
              - If a strategy DataFrame has a datetime index, reindex with method="ffill"
                to preserve the last known signal across candle timestamps.
              - If the strategy DataFrame does not have a datetime index but its length
                matches the candle matrix, align by position to preserve per-row signals.
              - Otherwise reindex to the candle index and fall back to deterministic fill.
              - Record per-strategy diagnostics (n_rows, nonzero_count, n_unique, pct_nonzero,
                sample_values) in self._last_strategy_feature_diagnostics for lineage/debugging.
            """
            if not strategies:
                self._last_strategy_feature_summary = {}
                self._last_feature_origin_map = {}
                self._last_strategy_feature_diagnostics = {}
                return features

            X = features.copy()
            summary: Dict[str, Dict[str, Dict[str, float]]] = {}
            feature_origin_map: Dict[str, Dict[str, str]] = {}
            diagnostics: Dict[str, Dict[str, object]] = {}

            for strat_name, strat_df in strategies.items():
                if strat_df is None or strat_df.empty:
                    continue

                df_s = strat_df.copy()

                # Normalize index if a timestamp column exists
                if "timestamp" in df_s.columns:
                    try:
                        df_s["timestamp"] = pd.to_datetime(df_s["timestamp"], utc=True)
                        df_s = df_s.sort_values("timestamp").set_index("timestamp")
                    except Exception:
                        # If conversion fails, continue with original index
                        df_s = df_s.copy()

                # Robust alignment to candle timestamps:
                #  - If df_s has a datetime index, forward-fill to propagate last known signal.
                #  - If df_s is positional and lengths match, align by position to preserve per-row signals.
                #  - Otherwise reindex to X.index (fallback).
                try:
                    if hasattr(df_s.index, "dtype") and pd.api.types.is_datetime64_any_dtype(df_s.index):
                        df_s = df_s.reindex(X.index, method="ffill")
                    else:
                        if len(df_s) == len(X):
                            # Align by position: set df_s index to match X's positional index
                            df_s = df_s.reset_index(drop=True)
                            temp_idx = X.reset_index(drop=True).index
                            df_s.index = temp_idx
                            df_s = df_s.reindex(temp_idx)
                        else:
                            df_s = df_s.reindex(X.index)
                except Exception:
                    df_s = df_s.reindex(X.index)

                # Replace inf/-inf then fill remaining NaNs with 0.0 deterministically
                df_s = df_s.replace([np.inf, -np.inf], np.nan).fillna(0.0)

                # Build summary for lineage
                strat_summary: Dict[str, Dict[str, float]] = {}

                for col in df_s.columns:
                    s = df_s[col]

                    # Namespaced feature name
                    namespaced = f"strategy_{strat_name}_{col}"
                    if namespaced in X.columns:
                        raise ValueError(
                            f"Feature collision: '{namespaced}' already exists in feature matrix"
                        )

                    # Add feature
                    X[namespaced] = s.values

                    # Record origin mapping
                    feature_origin_map[namespaced] = {"strategy": strat_name, "col": col}

                    # Summary stats
                    strat_summary[col] = {
                        "mean": float(s.mean()),
                        "std": float(s.std()),
                        "min": float(s.min()),
                        "max": float(s.max()),
                        "p25": float(s.quantile(0.25)),
                        "p50": float(s.quantile(0.50)),
                        "p75": float(s.quantile(0.75)),
                    }

                summary[strat_name] = strat_summary

                # Per-strategy diagnostics for lineage and debugging (best-effort)
                try:
                    n_rows = len(df_s)
                    nonzero_count = int(((df_s != 0).any(axis=1)).sum())
                    n_unique = {col: int(df_s[col].nunique(dropna=True)) for col in df_s.columns}
                    pct_nonzero = {col: float((df_s[col] != 0).mean()) for col in df_s.columns}
                    sample_values = {col: df_s[col].dropna().unique()[:5].tolist() for col in df_s.columns}

                    diagnostics[strat_name] = {
                        "n_rows": n_rows,
                        "nonzero_count": nonzero_count,
                        "n_unique": n_unique,
                        "pct_nonzero": pct_nonzero,
                        "sample_values": sample_values,
                    }
                except Exception:
                    # Non-fatal: diagnostics are best-effort
                    diagnostics.setdefault(strat_name, {"error": "diagnostics_failed"})

            # Store for training pipeline → lineage
            self._last_strategy_feature_summary = summary
            self._last_feature_origin_map = feature_origin_map
            self._last_strategy_feature_diagnostics = diagnostics

            return X

    # -------------------------------------------------------------------------
    # Regime features (from development FeatureBuilder (2))
    # -------------------------------------------------------------------------

    def _add_regime_features(
        self,
        features: pd.DataFrame,
        regimes: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Add regime label and confidence as features.

        - regime_label is one-hot encoded into regime_label_<value>
        - regime_confidence is kept as a numeric column if present

        Any missing timestamps are aligned to the features index; missing
        regimes are treated as "unknown" with 0 confidence.
        """
        if regimes is None or regimes.empty:
            return features

        X = features.copy()

        reg = regimes.reindex(X.index)

        # Regime label one-hot
        if "regime_label" in reg.columns:
            labels = reg["regime_label"].fillna("unknown").astype(str)
            dummies = pd.get_dummies(labels, prefix="regime_label")
            for col in dummies.columns:
                X[col] = dummies[col].values

        # Regime confidence
        if "regime_confidence" in reg.columns:
            conf = pd.to_numeric(reg["regime_confidence"], errors="coerce").fillna(0.0)
            X["regime_confidence"] = conf.values

        return X

    # -------------------------------------------------------------------------
    # Experiment-driven feature adjustments (from development FeatureBuilder (2))
    # -------------------------------------------------------------------------

    def _apply_experiment_feature_config(
        self,
        features: pd.DataFrame,
        experiment_config: Optional[Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Apply experiment-driven feature adjustments:

        Supported keys in experiment_config (all optional):

        - "drop_features": List[str]
            Drop these columns if present.

        - "add_lag_features": List[int]
            For each lag k, add lagged versions of key features:
                close_lag_k, ret_1_lag_k

        - "add_volatility_window": int
            Add an alternate rolling volatility window for ret_1:
                ret_1_vol_<window>
        """
        if not experiment_config:
            return features

        X = features.copy()
        cfg = experiment_config

        # Drop features
        drop = cfg.get("drop_features")
        if drop:
            to_drop = [col for col in drop if col in X.columns]
            if to_drop:
                X = X.drop(columns=to_drop)

        # Lag features
        lags = cfg.get("add_lag_features")
        if lags and isinstance(lags, (list, tuple)):
            for k in lags:
                try:
                    k_int = int(k)
                except (TypeError, ValueError):
                    continue

                if "close" in X.columns:
                    X[f"close_lag_{k_int}"] = X["close"].shift(k_int)

                if "ret_1" in X.columns:
                    X[f"ret_1_lag_{k_int}"] = X["ret_1"].shift(k_int)

        # Alternate volatility window
        vol_window = cfg.get("add_volatility_window")
        if vol_window and isinstance(vol_window, int) and vol_window > 1:
            if "ret_1" in X.columns:
                X[f"ret_1_vol_{vol_window}"] = (
                    X["ret_1"].rolling(
                        window=vol_window,
                        min_periods=max(3, vol_window // 4),
                    ).std()
                )

        return X

    # -------------------------------------------------------------------------
    # Validation (from development FeatureBuilder (2))
    # -------------------------------------------------------------------------

    @staticmethod
    def _validate(features: pd.DataFrame) -> None:
        """
        Validate final feature matrix:

        - no completely empty DataFrame
        - no entirely-NaN columns
        - no inf values
        - forward-fill and back-fill remaining NaNs; if still NaNs remain,
          raise an error

        This keeps behavior deterministic and surfaces data issues early.
        """
        if features is None or features.empty:
            raise ValueError("FeatureBuilder.validate: empty feature matrix")

        X = features.copy()

        # Drop columns that are entirely NaN
        all_nan_cols = [c for c in X.columns if X[c].isna().all()]
        if all_nan_cols:
            X = X.drop(columns=all_nan_cols)

        if X.empty:
            raise ValueError(
                "FeatureBuilder.validate: all feature columns were NaN after engineering"
            )

        # Replace inf / -inf with NaN
        X = X.replace([np.inf, -np.inf], np.nan)

        # Fill remaining NaNs with ffill/bfill
        X = X.ffill().bfill()

        if X.isna().any().any():
            bad_cols = [c for c in X.columns if X[c].isna().any()]
            raise ValueError(
                f"FeatureBuilder.validate: NaNs remain after ffill/bfill in columns: {bad_cols}"
            )

        # If validation passes, write back into original DataFrame in-place
        features.loc[:, X.columns] = X[X.columns]

    # -------------------------------------------------------------------------
    # RAW MATH TECHNICAL INDICATOR ENGINE (from development FeatureBuilder (2))
    # -------------------------------------------------------------------------

    def _build_raw_math_features(self, candles: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate a set of raw technical indicators for use in feature engineering.

        Returns a DataFrame aligned with candles index.
        """
        df = candles.copy()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").set_index("timestamp")

        # 1. Momentum Indicators
        df["rsi_raw"] = self._calculate_rsi(df["close"], period=14)
        df["roc_10"] = df["close"].pct_change(periods=10)

        # 2. Volatility Indicators
        df["atr_raw"] = self._calculate_atr(df, period=14)
        df["bb_width_20"] = self._calculate_bb_width(df["close"], period=20)

        # 3. Trend Indicators (Raw Math only)
        df["ema_fast_12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema_slow_26"] = df["close"].ewm(span=26, adjust=False).mean()
        df["macd_12_26"] = df["ema_fast_12"] - df["ema_slow_26"]

        # 4. Volume Features
        df["vol_sma_20"] = df["volume"].rolling(window=20).mean()
        df["relative_vol_20"] = df["volume"] / (df["vol_sma_20"] + 1e-9)

        # 5. Price Action Geometry
        df["body_size_raw"] = (df["close"] - df["open"]).abs()
        df["upper_wick_raw"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_wick_raw"] = df[["open", "close"]].min(axis=1) - df["low"]

        return df

    # --- Internal Math Helpers ---

    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=period).mean()

    def _calculate_bb_width(self, series: pd.Series, period: int) -> pd.Series:
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return (upper - lower) / (sma + 1e-9)

    # -------------------------------------------------------------------------
    # Feature catalog
    # -------------------------------------------------------------------------

    def _build_feature_catalog(self) -> Dict[str, str]:
        """
        This is the distilled, static catalog of features, mirroring the intent of
        modules/ui/explain_panel.FEATURE_CATALOG and the knowledge graph.
        """
        return {
            "body": "Absolute candle body (close - open).",
            "range": "High-low range of the candle.",
            "upper_wick": "Distance from max(open, close) to high.",
            "lower_wick": "Distance from low to min(open, close).",
            "body_pct": "Candle body as a fraction of open.",
            "range_pct": "Candle range as a fraction of open.",
            "atr": "Average True Range over configured window.",
            "atr_pct": "ATR as a fraction of close.",
            f"ma_{self.config.fast_ma}": f"Fast moving average ({self.config.fast_ma} bars).",
            f"ma_{self.config.slow_ma}": f"Slow moving average ({self.config.slow_ma} bars).",
            "ma_spread": "Difference between fast and slow moving averages.",
            "ma_spread_pct": "MA spread as a fraction of slow MA.",
            "rsi": "Relative Strength Index over configured window.",
            "volume_ma": "Rolling mean of volume.",
            "volume_zscore": "Z-score of volume relative to rolling window.",
            "volatility": "Rolling standard deviation of returns.",
            "vol_regime": "Discrete volatility regime (0=low,1=medium,2=high).",
            "drift_mean": "Rolling mean of returns over drift window.",
            "drift_std": "Rolling std of returns over drift window.",
            "drift_score": "Drift mean divided by drift std.",
            "sentiment_score": "Raw sentiment score aligned to candles.",
            "sentiment_ma": "Rolling mean of sentiment score.",
            "sentiment_zscore": "Z-score of sentiment relative to rolling window.",
            "drift_sentiment_adjusted": "Drift score adjusted by sentiment mean.",
            "mtf_confirmation_score": "Average direction across higher timeframes.",
            "direction": "Sign of close-to-close change.",
            "dir_sum_w": "Rolling sum of direction over sequencing window.",
            "dir_up_count": "Count of up candles in sequencing window.",
            "dir_down_count": "Count of down candles in sequencing window.",
            "momentum_burst": "Flag for strong directional burst in recent candles.",
        }

    # -------------------------------------------------------------------------
    # Snapshot getters for orchestration / lineage
    # -------------------------------------------------------------------------

    def get_last_strategy_feature_summary(self) -> Dict[str, Any]:
        """
        Return the last computed strategy feature summary (empty dict if none).
        """
        return getattr(self, "_last_strategy_feature_summary", {})

    def get_feature_origin_map(self) -> Dict[str, Dict[str, str]]:
        """
        Return a mapping from namespaced feature -> {"strategy": name, "col": original_col}.
        """
        return getattr(self, "_last_feature_origin_map", {})

    def get_last_strategy_feature_diagnostics(self) -> Dict[str, Any]:
        """
        Return the last per-strategy diagnostics recorded during feature building.
        Diagnostics include n_rows, nonzero_count, n_unique, pct_nonzero, and sample_values.
        """
        return getattr(self, "_last_strategy_feature_diagnostics", {})
