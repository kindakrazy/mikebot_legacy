# C:\mikebot\minions\regime_switcher.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regime state
# ---------------------------------------------------------------------------

@dataclass
class RegimeState:
    regime_id: Optional[int] = None
    regime_score: float = 0.0
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RegimeSwitcherConfig:
    """
    Controls regime persistence, hysteresis, and confidence gating.

    This is the mikebot v3 distillation of:
      - HighstrikeSignals/modules/regime_switcher.py
      - HighstrikeSignals/modules/hybrid_personality.py
      - clustering_scout regime metadata
    """

    # Minimum number of consecutive updates before switching
    min_persistence: int = 50

    # Minimum score delta required to switch regimes
    hysteresis: float = 0.1

    # Minimum confidence required to consider a regime update
    min_confidence: float = 0.5


# ---------------------------------------------------------------------------
# Regime switcher
# ---------------------------------------------------------------------------

class RegimeSwitcher:
    """
    Regime persistence + hysteresis + confidence gating.

    Intended to be fed by clustering_scout and queried by:
      - PersonalityManager
      - Multi-agent engine
      - Feature builder (for regime features)
    """

    def __init__(self, config: RegimeSwitcherConfig) -> None:
        self.config = config
        self._current = RegimeState()
        self._candidate = RegimeState()
        self._age = 0

    # ----------------------------------------------------------------------
    # Update logic
    # ----------------------------------------------------------------------

    def update(self, regime_id: int, regime_score: float, confidence: float) -> None:
        """
        Called by clustering_scout when it has a new regime estimate.
        """
        # Confidence gating
        if confidence < self.config.min_confidence:
            return

        # First regime ever
        if self._current.regime_id is None:
            self._current = RegimeState(regime_id, regime_score, confidence)
            self._age = 0
            return

        # Same regime → refresh score/confidence
        if regime_id == self._current.regime_id:
            self._current.regime_score = regime_score
            self._current.confidence = confidence
            self._age += 1
            return

        # Different regime → candidate
        self._candidate = RegimeState(regime_id, regime_score, confidence)
        self._age += 1

        # Hysteresis + persistence
        score_delta = abs(self._candidate.regime_score - self._current.regime_score)

        if (
            self._age >= self.config.min_persistence
            and score_delta >= self.config.hysteresis
        ):
            logger.info(
                "RegimeSwitcher: switching regime %s → %s (Δ=%.3f)",
                self._current.regime_id,
                self._candidate.regime_id,
                score_delta,
            )
            self._current = self._candidate
            self._age = 0

    # ----------------------------------------------------------------------
    # Accessors
    # ----------------------------------------------------------------------

    def get_state(self) -> RegimeState:
        return self._current

    def snapshot(self) -> Dict[str, Any]:
        return {
            "regime_id": self._current.regime_id,
            "regime_score": self._current.regime_score,
            "confidence": self._current.confidence,
            "age": self._age,
        }
