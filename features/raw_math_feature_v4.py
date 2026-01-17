from __future__ import annotations

from typing import Any, Dict, Sequence, Optional

from mikebot.data.candle_schema import Candle
from mikebot.features.feature_interface import Feature, FeatureState


class RawMathFeature(Feature):
    """
    Compact raw-math indicator set, mirroring the spirit of the
    development-era raw math engine in v3.
    """

    @property
    def name(self) -> str:
        return "raw_math"

    @property
    def requires_lookback(self) -> int:
        return 26  # enough for MACD-like spans

    def compute(
        self,
        candle: Candle,
        history: Sequence[Candle],
        state: Optional[FeatureState],
    ) -> Dict[str, Any]:
        candles = list(history) + [candle]
        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        volumes = [c.volume for c in candles]

        # RSI-like raw
        rsi_raw = self._rsi(closes, period=14)
        # ROC 10
        roc_10 = self._roc(closes, period=10)
        # ATR-like raw
        atr_raw = self._atr(highs, lows, closes, period=14)
        # Bollinger width 20
        bb_width_20 = self._bb_width(closes, period=20)
        # EMA fast/slow + MACD
        ema_fast_12 = self._ema(closes, span=12)
        ema_slow_26 = self._ema(closes, span=26)
        macd_12_26 = ema_fast_12 - ema_slow_26
        # Volume SMA 20 + relative volume
        vol_sma_20 = self._sma(volumes, window=20)
        relative_vol_20 = volumes[-1] / vol_sma_20 if vol_sma_20 != 0 else 0.0
        # Price action geometry (raw)
        body_size_raw = abs(candle.close - candle.open)
        upper_wick_raw = candle.high - max(candle.open, candle.close)
        lower_wick_raw = min(candle.open, candle.close) - candle.low

        return {
            "rsi_raw": rsi_raw,
            "roc_10": roc_10,
            "atr_raw": atr_raw,
            "bb_width_20": bb_width_20,
            "ema_fast_12": ema_fast_12,
            "ema_slow_26": ema_slow_26,
            "macd_12_26": macd_12_26,
            "vol_sma_20": vol_sma_20,
            "relative_vol_20": relative_vol_20,
            "body_size_raw": body_size_raw,
            "upper_wick_raw": upper_wick_raw,
            "lower_wick_raw": lower_wick_raw,
        }

    # --- helpers ---

    def _sma(self, values: Sequence[float], window: int) -> float:
        if not values:
            return 0.0
        if len(values) < window:
            subset = values
        else:
            subset = values[-window:]
        return sum(subset) / len(subset)

    def _ema(self, values: Sequence[float], span: int) -> float:
        if not values:
            return 0.0
        alpha = 2 / (span + 1)
        ema = values[0]
        for v in values[1:]:
            ema = alpha * v + (1 - alpha) * ema
        return ema

    def _rsi(self, closes: Sequence[float], period: int) -> float:
        if len(closes) < period + 1:
            return 50.0
        gains: list[float] = []
        losses: list[float] = []
        for i in range(1, len(closes)):
            delta = closes[i] - closes[i - 1]
            if delta > 0:
                gains.append(delta)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(-delta)
        gains = gains[-period:]
        losses = losses[-period:]
        avg_gain = sum(gains) / len(gains) if gains else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _roc(self, closes: Sequence[float], period: int) -> float:
        if len(closes) <= period:
            return 0.0
        prev = closes[-period - 1]
        cur = closes[-1]
        return (cur - prev) / prev if prev != 0 else 0.0

    def _atr(
        self,
        highs: Sequence[float],
        lows: Sequence[float],
        closes: Sequence[float],
        period: int,
    ) -> float:
        if len(highs) < 2:
            return 0.0
        trs: list[float] = []
        for i in range(1, len(highs)):
            h = highs[i]
            l = lows[i]
            prev_close = closes[i - 1]
            tr1 = h - l
            tr2 = abs(h - prev_close)
            tr3 = abs(l - prev_close)
            trs.append(max(tr1, tr2, tr3))
        if len(trs) < period:
            subset = trs
        else:
            subset = trs[-period:]
        return sum(subset) / len(subset) if subset else 0.0

    def _bb_width(self, closes: Sequence[float], period: int) -> float:
        if len(closes) < period:
            return 0.0
        subset = closes[-period:]
        mean = sum(subset) / len(subset)
        var = sum((c - mean) ** 2 for c in subset) / len(subset)
        std = var ** 0.5
        upper = mean + 2 * std
        lower = mean - 2 * std
        return (upper - lower) / mean if mean != 0 else 0.0