import pandas as pd
import numpy as np


class StrategyImpl:
    """
    Modern Bear Flag strategy.
    Produces continuous ML-friendly features instead of binary signals.
    """

    name = "BearFlag"
    version = "1.0"

    # Parameter names used by the optimizer and UI
    parameters = [
        "flagpole_min_atr",
        "pullback_min_pct",
        "pullback_max_pct",
        "breakout_strength_min",
        "volume_ratio_min",
    ]

    def __init__(self, config: dict):
        self.cfg = config

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------

    @staticmethod
    def _atr(df, window=14):
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(window).mean()

    # ------------------------------------------------------------
    # Core compute
    # ------------------------------------------------------------

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute ML-friendly Bear Flag features.
        Returns a DataFrame with:
            - signal
            - flagpole_strength
            - pullback_depth
            - breakout_strength
            - volume_ratio
        """

        atr = self._atr(df)

        # --------------------------------------------------------
        # 1. Flagpole strength (big bearish impulse)
        # --------------------------------------------------------
        # Look back 3 bars for the pole
        flagpole = (df["open"].shift(3) - df["close"].shift(3)) / atr.shift(3)
        flagpole_ok = flagpole > self.cfg.get("flagpole_min_atr", 1.5)

        # --------------------------------------------------------
        # 2. Pullback depth (controlled upward drift)
        # --------------------------------------------------------
        pullback = (df["close"].shift(2) - df["close"].shift(3)) / atr.shift(3)
        pullback_ok = pullback.between(
            self.cfg.get("pullback_min_pct", 0.25),
            self.cfg.get("pullback_max_pct", 0.6)
        )

        # --------------------------------------------------------
        # 3. Breakout strength (downward continuation)
        # --------------------------------------------------------
        breakout_strength = (df["low"].shift(2) - df["close"]) / atr
        breakout_ok = breakout_strength > self.cfg.get("breakout_strength_min", 0.1)

        # --------------------------------------------------------
        # 4. Volume confirmation
        # --------------------------------------------------------
        vol_ratio = df["volume"] / df["volume"].rolling(20).mean()
        vol_ok = vol_ratio > self.cfg.get("volume_ratio_min", 1.2)

        # --------------------------------------------------------
        # 5. Final continuous signal
        # --------------------------------------------------------
        # Instead of a binary 0/1, we produce a continuous score:
        #   -1.0 = strong bearish flag
        #    0.0 = no pattern
        #
        # This is far more useful for ML.
        score = (
            flagpole_ok.astype(float)
            * pullback_ok.astype(float)
            * breakout_ok.astype(float)
            * vol_ok.astype(float)
        )

        signal = -1.0 * score

        return pd.DataFrame({
            "signal": signal.fillna(0.0),
            "flagpole_strength": flagpole.fillna(0.0),
            "pullback_depth": pullback.fillna(0.0),
            "breakout_strength": breakout_strength.fillna(0.0),
            "volume_ratio": vol_ratio.fillna(0.0),
        })
