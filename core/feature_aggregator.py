# mikebot/core/feature_aggregator.py

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np

from mikebot.core.feature_builder import FeatureBuilder
from mikebot.core.regime_detector import RegimeDetector, MarketRegime

logger = logging.getLogger(__name__)


class FeatureAggregator:
    """Consolidates disparate data streams into unified feature matrices.

    Responsibilities:
    - Fuse raw technical features (from FeatureBuilder) with:
        * Market regime context (from RegimeDetector)
        * Strategy metadata / context
    - Provide both single-timeframe and multi-timeframe aggregation paths.
    - Ensure final feature matrices are numerically safe for modeling.

    Typical uses:
    - Live inference: build a single-row feature vector with context.
    - Batch experiments: build historical feature blocks for MetaTrainer.
    - Multi-timeframe research: merge M1/M5/M15 features and analyze correlations.
    """

    def __init__(self, feature_builder: FeatureBuilder, regime_detector: RegimeDetector):
        self.fb = feature_builder
        self.rd = regime_detector

    # ---------------------------------------------------------------------
    # SINGLE-TIMEFRAME AGGREGATION
    # ---------------------------------------------------------------------

    def aggregate_live(self, raw_data: pd.DataFrame, strategy_context: Dict[str, Any]) -> pd.DataFrame:
        """
        Create a single-row feature vector for live inference.

        Parameters
        ----------
        raw_data:
            Recent candle/feature window (DataFrame) for a single symbol/timeframe.

        strategy_context:
            Dictionary containing strategy parameters / metadata. Expected shape:
            {
                "params": {
                    "some_param": 1.0,
                    ...
                },
                ...
            }

        Returns
        -------
        A 1-row DataFrame with:
            - Engineered features
            - Regime context
            - Strategy context
        """
        features = self.fb.build_live(raw_data)
        regime_info = self.rd.detect(raw_data)
        return self._fuse(features, regime_info, strategy_context)

    def aggregate_batch(self, raw_data: pd.DataFrame, strategy_id: str) -> pd.DataFrame:
        """
        Process a historical block of data for MetaTrainer experiments.

        Parameters
        ----------
        raw_data:
            Full historical window for a single symbol/timeframe.

        strategy_id:
            Identifier for strategy configuration (not used directly yet, but
            kept for future extension, e.g. loading strategy-specific context).

        Returns
        -------
        DataFrame with:
            - Engineered features
            - Regime one-hot columns (is_regime_X)
        """
        features = self.fb.build_batch(raw_data)
        regime_map = self.rd.get_regime_history(raw_data)

        combined = pd.concat([features, regime_map], axis=1).dropna()
        combined = pd.get_dummies(combined, columns=["regime"], prefix="is_regime")

        for r in MarketRegime:
            col = f"is_regime_{r.value}"
            if col not in combined.columns:
                combined[col] = 0

        return combined

    def _fuse(
        self,
        features: pd.DataFrame,
        regime_info: Dict[str, Any],
        strategy_context: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Internal logic to merge features with regime + strategy context.
        """
        X = features.copy()

        # Regime metrics (continuous context)
        for key, val in regime_info.get("metrics", {}).items():
            X[f"context_{key}"] = val

        # Regime one-hot flags for current regime
        current_regime = regime_info.get("regime_id", "unstable")
        for r in MarketRegime:
            X[f"is_regime_{r.value}"] = 1 if r.value == current_regime else 0

        # Strategy numeric parameters
        for key, val in strategy_context.get("params", {}).items():
            if isinstance(val, (int, float)):
                X[f"strat_param_{key}"] = val

        return self.validate(X)

    # ---------------------------------------------------------------------
    # MULTI-TIMEFRAME AGGREGATION
    # ---------------------------------------------------------------------

    def merge_timeframes(self, tf_features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Align and merge multiple timeframe feature tables.

        Uses the numerically smallest timeframe (e.g., "1" < "5" < "15") as the
        base index. Higher timeframes are forward-filled to align with the base
        timeframe index.

        Parameters
        ----------
        tf_features:
            Mapping from timeframe string -> features DataFrame, e.g.:
            {
                "1": df_M1_features,
                "5": df_M5_features,
                "15": df_M15_features,
            }

            Each DataFrame is expected to be indexed by timestamp.

        Returns
        -------
        DataFrame where:
            - Index is the base timeframe index.
            - Columns are suffixed with _<timeframe>, e.g. close_1, close_5, ...
        """
        if not tf_features:
            raise ValueError("FeatureAggregator.merge_timeframes: no timeframe features provided")

        # Determine base timeframe by numeric value of the key
        try:
            base_tf = sorted(tf_features.keys(), key=lambda x: int(x))[0]
        except ValueError as e:
            raise ValueError(
                f"FeatureAggregator.merge_timeframes: timeframe keys must be numeric strings, got {list(tf_features.keys())}"
            ) from e

        base = tf_features[base_tf].copy()
        if base.empty:
            raise ValueError(f"FeatureAggregator.merge_timeframes: base timeframe '{base_tf}' features are empty")

        # Start from base timeframe, rename columns
        merged = base.copy()
        merged.columns = [f"{c}_{base_tf}" for c in merged.columns]

        # Merge higher timeframes
        for tf, df in tf_features.items():
            if tf == base_tf:
                continue
            if df is None or df.empty:
                logger.warning("FeatureAggregator.merge_timeframes: timeframe '%s' has empty features, skipping", tf)
                continue

            df2 = df.copy()
            df2.columns = [f"{c}_{tf}" for c in df2.columns]

            # Align on base index with forward-fill
            df2 = df2.reindex(merged.index, method="ffill")
            merged = pd.concat([merged, df2], axis=1)

        # Drop any rows that are still incomplete
        merged = merged.dropna(how="any")
        return merged

    def aggregate_multi_tf(self, tf_features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Full multi-timeframe aggregation pipeline.

        Parameters
        ----------
        tf_features:
            Mapping timeframe -> features DataFrame.

        Returns
        -------
        Validated, merged multi-TF feature matrix.
        """
        merged = self.merge_timeframes(tf_features)
        merged = self.validate(merged)
        return merged

    def compute_cross_tf_correlations(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Pearson correlations across all merged TF features.

        Parameters
        ----------
        merged_df:
            Output of aggregate_multi_tf or merge_timeframes.

        Returns
        -------
        Correlation matrix (DataFrame).
        """
        if merged_df.empty:
            raise ValueError("FeatureAggregator.compute_cross_tf_correlations: input DataFrame is empty")
        return merged_df.corr()

    def compute_lagged_correlations(
        self,
        merged_df: pd.DataFrame,
        max_lag: int = 10,
    ) -> Dict[int, pd.Series]:
        """
        Compute lagged correlations for each lag from -max_lag to +max_lag.

        For each lag L, we compute corr(X_t, X_{t+L}) via corrwith on shifted data.

        Parameters
        ----------
        merged_df:
            Output of aggregate_multi_tf or merge_timeframes.

        max_lag:
            Maximum absolute lag to consider.

        Returns
        -------
        Dict[int, pd.Series] where:
            - key: lag (negative, zero, positive)
            - value: correlation of each column with its lagged version
        """
        if merged_df.empty:
            raise ValueError("FeatureAggregator.compute_lagged_correlations: input DataFrame is empty")

        results: Dict[int, pd.Series] = {}
        for lag in range(-max_lag, max_lag + 1):
            shifted = merged_df.shift(lag)
            results[lag] = merged_df.corrwith(shifted)
        return results

    # ---------------------------------------------------------------------
    # VALIDATION / SANITY
    # ---------------------------------------------------------------------

    @staticmethod
    def validate(features: pd.DataFrame) -> pd.DataFrame:
        """
        Final data integrity check before passing to a model.

        - Rejects empty frames.
        - Replaces inf/-inf with NaN.
        - Fills interior NaNs via forward/backward fill.
        - Fills any remaining NaNs with 0.

        This keeps downstream models numerically stable.
        """
        if features is None or features.empty:
            raise ValueError("FeatureAggregator.validate: resulting feature matrix is empty.")

        X = features.replace([np.inf, -np.inf], np.nan)
        X = X.ffill().bfill()

        if X.isna().any().any():
            X = X.fillna(0)

        return X
