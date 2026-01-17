# mikebot/features/multi_tf_aggregator_v4.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


@dataclass
class MultiTFConfigV4:
    """
    Configuration for multi-timeframe aggregation.

    Example:
        MultiTFConfigV4(
            targets=[
                ("M5", 5),
                ("M15", 15),
                ("H1", 60),
            ]
        )
    """
    targets: Sequence[Tuple[str, int]]  # (tf_name, tf_minutes)


class MultiTFAggregatorV4:
    """
    Streaming-friendly multi-timeframe aggregator for v4.

    Responsibilities:
    - Accept a base timeframe candle stream (e.g., M1 or M5)
    - Maintain rolling buffers for higher timeframes
    - Emit aggregated candles for each configured TF
    - Return a dict {tf_name: aggregated_df}

    This module is intentionally simple and deterministic.
    """

    def __init__(self, config: MultiTFConfigV4) -> None:
        self.config = config
        self._buffers: Dict[str, List[pd.Series]] = {
            tf_name: [] for tf_name, _ in config.targets
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, candle: pd.Series) -> Dict[str, Optional[pd.Series]]:
        """
        Update all timeframe buffers with a new base candle.

        Returns:
            {tf_name: aggregated_candle_or_None}
        """
        out: Dict[str, Optional[pd.Series]] = {}

        for tf_name, tf_minutes in self.config.targets:
            buf = self._buffers[tf_name]
            buf.append(candle)

            # If we have enough candles to form a full TF block
            if len(buf) == tf_minutes:
                aggregated = self._aggregate_block(buf)
                out[tf_name] = aggregated
                self._buffers[tf_name] = []
            else:
                out[tf_name] = None

        return out

    def aggregate_history(
        self,
        candles: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """
        Offline/batch aggregation for historical data.

        Returns:
            {tf_name: aggregated_df}
        """
        result: Dict[str, pd.DataFrame] = {}

        for tf_name, tf_minutes in self.config.targets:
            blocks = []
            buf: List[pd.Series] = []

            for _, row in candles.iterrows():
                buf.append(row)
                if len(buf) == tf_minutes:
                    blocks.append(self._aggregate_block(buf))
                    buf = []

            if blocks:
                df = pd.DataFrame(blocks)
                df.index = range(len(df))
                result[tf_name] = df
            else:
                result[tf_name] = pd.DataFrame()

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _aggregate_block(self, block: Sequence[pd.Series]) -> pd.Series:
        """
        Aggregate a block of candles into a single higher-timeframe candle.

        Rules:
        - open: first open
        - high: max high
        - low: min low
        - close: last close
        - volume: sum volume (if present)
        """
        opens = block[0]["open"]
        highs = max(c["high"] for c in block)
        lows = min(c["low"] for c in block)
        closes = block[-1]["close"]

        volume = None
        if "volume" in block[0]:
            volume = sum(float(c.get("volume", 0.0)) for c in block)

        data = {
            "open": float(opens),
            "high": float(highs),
            "low": float(lows),
            "close": float(closes),
        }

        if volume is not None:
            data["volume"] = float(volume)

        return pd.Series(data)