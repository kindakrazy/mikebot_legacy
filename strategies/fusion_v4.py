# mikebot/strategies/fusion_v4.py

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FusionDecision:
    """Final fused decision for the orchestrator."""
    selected_strategy: str
    signal: float
    confidence: float
    all_signals: Dict[str, float]
    reason: str
    performance_snapshot: Dict[str, Any]


class StrategyFusionV4:
    """
    Winner‑take‑all multi‑strategy fusion layer.

    Responsibilities:
      - Accept raw strategy signals
      - Compute confidence = abs(signal)
      - Track rolling performance per strategy
      - Select the most confident strategy
      - Return a single fused decision

    This layer does NOT:
      - override strategy signals
      - apply thresholds
      - apply regime logic
      - blend signals
      - suppress trades
    """

    def __init__(self, performance_window: int = 200):
        # Rolling performance tracking per strategy
        self.performance_window = performance_window
        self._perf: Dict[str, list] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(self, signals: Dict[str, float]) -> Optional[FusionDecision]:
        """
        Main entry point.

        Parameters
        ----------
        signals : {strategy_id: signal_value}

        Returns
        -------
        FusionDecision or None
        """
        if not signals:
            return None

        # Compute confidence = abs(signal)
        confidences = {sid: abs(sig) for sid, sig in signals.items()}

        # Winner‑take‑all
        winner = max(confidences, key=lambda sid: confidences[sid])
        winner_signal = signals[winner]
        winner_conf = confidences[winner]

        # Update rolling performance
        self._update_performance(winner, winner_conf)

        decision = FusionDecision(
            selected_strategy=winner,
            signal=winner_signal,
            confidence=winner_conf,
            all_signals=dict(signals),
            reason=f"winner_take_all: strongest absolute signal ({winner_conf:.4f})",
            performance_snapshot=self._performance_snapshot(),
        )

        return decision

    # ------------------------------------------------------------------
    # Performance tracking
    # ------------------------------------------------------------------

    def _update_performance(self, strategy_id: str, confidence: float) -> None:
        """Track rolling confidence history per strategy."""
        if strategy_id not in self._perf:
            self._perf[strategy_id] = []

        buf = self._perf[strategy_id]
        buf.append(confidence)

        # Trim to window
        if len(buf) > self.performance_window:
            self._perf[strategy_id] = buf[-self.performance_window:]

    def _performance_snapshot(self) -> Dict[str, Any]:
        """Return summary stats for each strategy."""
        snap = {}
        for sid, hist in self._perf.items():
            if not hist:
                continue
            s = pd.Series(hist)
            snap[sid] = {
                "count": int(len(hist)),
                "mean_conf": float(s.mean()),
                "max_conf": float(s.max()),
                "min_conf": float(s.min()),
                "std_conf": float(s.std() if len(hist) > 1 else 0.0),
            }
        return snap