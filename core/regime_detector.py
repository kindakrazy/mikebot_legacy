# mikebot/core/regime_detector.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any
from enum import Enum


class MarketRegime(Enum):
    TREND_BULL = "trend_bull"
    TREND_BEAR = "trend_bear"
    RANGING_LOW_VOL = "ranging_low_vol"
    RANGING_HIGH_VOL = "ranging_high_vol"
    UNSTABLE = "unstable"


class RegimeDetector:
    """
    Categorizes market conditions to provide context for the Self-Learning system.
    Used by MetaTrainer and the V4 cockpit.
    """

    def __init__(self, window: int = 14):
        self.window = window

    # ------------------------------------------------------------------
    # Core detection logic
    # ------------------------------------------------------------------

    def detect(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the most recent window of OHLC data to determine regime.

        Returns:
            {
                "regime_id": <MarketRegime.value>,
                "metrics": {
                    "volatility": float,
                    "trend_slope": float,
                    "efficiency_ratio": float
                },
                "timestamp": <iso8601 or str>
            }
        """
        if len(df) < self.window:
            return self._default_regime()

        # 1. Volatility (std of log returns)
        returns = np.log(df["close"] / df["close"].shift(1))
        volatility = returns.tail(self.window).std() * np.sqrt(self.window)

        # 2. Trend strength (EMA slope)
        ema = df["close"].ewm(span=self.window).mean()
        slope = (ema.iloc[-1] - ema.iloc[-self.window]) / ema.iloc[-self.window]

        # 3. Efficiency Ratio (ER)
        net_move = abs(df["close"].iloc[-1] - df["close"].iloc[-self.window])
        sum_abs_moves = df["close"].diff().abs().tail(self.window).sum()
        er = net_move / sum_abs_moves if sum_abs_moves != 0 else 0.0

        # 4. Regime assignment
        regime = MarketRegime.UNSTABLE

        if er > 0.6:  # trending
            regime = MarketRegime.TREND_BULL if slope > 0 else MarketRegime.TREND_BEAR

        elif er < 0.3:  # ranging
            # FIX: volatility is a scalar, so compare to its own magnitude
            # (previous code incorrectly used volatility.mean())
            if volatility > 0.02:  # threshold can be tuned
                regime = MarketRegime.RANGING_HIGH_VOL
            else:
                regime = MarketRegime.RANGING_LOW_VOL

        else:
            regime = MarketRegime.RANGING_LOW_VOL

        ts = df.index[-1]
        if isinstance(ts, pd.Timestamp):
            ts = ts.isoformat()

        return {
            "regime_id": regime.value,
            "metrics": {
                "volatility": float(volatility),
                "trend_slope": float(slope),
                "efficiency_ratio": float(er),
            },
            "timestamp": str(ts),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _default_regime(self) -> Dict[str, Any]:
        return {
            "regime_id": MarketRegime.UNSTABLE.value,
            "metrics": {
                "volatility": 0.0,
                "trend_slope": 0.0,
                "efficiency_ratio": 0.0,
            },
            "timestamp": "N/A",
        }

    def get_regime_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a historical regime map for the entire dataframe.
        """
        regimes = []

        for i in range(self.window, len(df)):
            window_df = df.iloc[i - self.window : i]
            res = self.detect(window_df)
            regimes.append(
                {
                    "timestamp": df.index[i],
                    "regime": res["regime_id"],
                    **res["metrics"],
                }
            )

        return pd.DataFrame(regimes).set_index("timestamp")


# ---------------------------------------------------------------------------
# Global accessor for RegimeDetector (V4 cockpit)
# ---------------------------------------------------------------------------

_regime_detector_global = None


def set_global_regime_detector(detector) -> None:
    """
    Register the process-wide RegimeDetector instance.
    Call this once during startup.
    """
    global _regime_detector_global
    _regime_detector_global = detector


def get_global_regime_detector():
    """
    Return the process-wide RegimeDetector instance, or None if not set.
    """
    return _regime_detector_global
