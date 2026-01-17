# mikebot/minions/personality.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Personality definition
# ---------------------------------------------------------------------------

@dataclass
class PersonalityProfile:
    """
    A personality profile that modulates multi-agent blending.
    """

    name: str

    aggression: float = 1.0
    caution: float = 1.0
    confidence_weight: float = 1.0
    noise_tolerance: float = 0.5
    regime_sensitivity: float = 1.0
    allow_evolution: bool = True


# ---------------------------------------------------------------------------
# Built-in personalities
# ---------------------------------------------------------------------------

DEFAULT_PERSONALITIES: Dict[str, PersonalityProfile] = {
    "neutral": PersonalityProfile(
        name="neutral",
        aggression=1.0,
        caution=1.0,
        confidence_weight=1.0,
        noise_tolerance=0.5,
        regime_sensitivity=1.0,
    ),
    "aggressive": PersonalityProfile(
        name="aggressive",
        aggression=1.6,
        caution=0.8,
        confidence_weight=1.2,
        noise_tolerance=0.7,
        regime_sensitivity=1.4,
    ),
    "conservative": PersonalityProfile(
        name="conservative",
        aggression=0.7,
        caution=1.4,
        confidence_weight=0.8,
        noise_tolerance=0.3,
        regime_sensitivity=0.8,
    ),
    "scout": PersonalityProfile(
        name="scout",
        aggression=1.2,
        caution=0.9,
        confidence_weight=1.5,
        noise_tolerance=0.6,
        regime_sensitivity=1.8,
    ),
}


# ---------------------------------------------------------------------------
# Personality manager
# ---------------------------------------------------------------------------

class PersonalityManager:
    """
    The personality engine.
    """

    def __init__(
        self,
        default: str = "neutral",
        personalities: Optional[Dict[str, PersonalityProfile]] = None,
    ) -> None:
        self.personalities = personalities or DEFAULT_PERSONALITIES
        self.active = self.personalities.get(default, DEFAULT_PERSONALITIES["neutral"])

        self._evolution_counter = 0

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def get_active(self) -> PersonalityProfile:
        return self.active

    def set_active(self, name: str) -> None:
        """
        Switch personality only if it actually changes.
        Prevents log spam.
        """
        if name not in self.personalities:
            return

        current = self.active.name
        if name != current:
            logger.info("PersonalityManager: switching to '%s'", name)
            self.active = self.personalities[name]

    def update_from_regime(self, regime_id: Optional[int], regime_score: Optional[float]) -> None:
        """
        Map regime → personality, but only trigger a switch when the
        resulting personality differs from the current one.
        """
        if regime_id is None or regime_score is None:
            return

        shift = np.clip(regime_score * self.active.regime_sensitivity, -1.0, 1.0)

        if shift > 0.2:
            target = "aggressive"
        elif shift < -0.2:
            target = "conservative"
        else:
            target = "neutral"

        if target != self.active.name:
            self.set_active(target)

    def evolve(self, loop_iteration: int) -> None:
        if not self.active.allow_evolution:
            return

        if loop_iteration % 500 != 0:
            return

        self._evolution_counter += 1

        delta = (np.random.rand() - 0.5) * 0.1
        self.active.aggression = float(np.clip(self.active.aggression + delta, 0.5, 2.0))

        delta = (np.random.rand() - 0.5) * 0.1
        self.active.caution = float(np.clip(self.active.caution + delta, 0.5, 2.0))

        logger.info(
            "PersonalityManager: evolved '%s' → aggression=%.2f, caution=%.2f",
            self.active.name,
            self.active.aggression,
            self.active.caution,
        )
    # ----------------------------------------------------------------------
    # Metadata
    # ----------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        p = self.active
        return {
            "name": p.name,
            "aggression": p.aggression,
            "caution": p.caution,
            "confidence_weight": p.confidence_weight,
            "noise_tolerance": p.noise_tolerance,
            "regime_sensitivity": p.regime_sensitivity,
            "allow_evolution": p.allow_evolution,
        }


# ---------------------------------------------------------------------------
# Global accessor for PersonalityManager (V4 cockpit)
# ---------------------------------------------------------------------------

_personality_manager_global = None


def set_global_personality_manager(manager) -> None:
    """
    Register the process-wide PersonalityManager instance.
    """
    global _personality_manager_global
    _personality_manager_global = manager


def get_global_personality_manager():
    """
    Return the process-wide PersonalityManager instance, or None if not set.
    """
    return _personality_manager_global
