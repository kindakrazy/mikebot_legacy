# C:\mikebot\minions\minion_decision.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass(frozen=True)
class MinionDecision:
    """
    Modern, explicit, typed decision returned by every minion.

    This replaces the legacy loosely-structured decision objects with a
    stable, predictable, validated structure that the orchestrator and
    multi-agent vote can rely on.
    """

    # Required fields
    minion_name: str
    action: str               # e.g. "long", "short", "hold", "reduce", "exit"
    score: float              # normalized decision strength (0–1 or -1–1)
    confidence: float         # model confidence (0–1)

    # Optional metadata (model-specific details, debug info, regime IDs, etc.)
    meta: Dict[str, Any] = field(default_factory=dict)

    # Optional symbol this decision applies to (if minion is symbol-specific)
    symbol: Optional[str] = None

    # Convenience helpers
    def is_hold(self) -> bool:
        return self.action == "hold"

    def is_entry(self) -> bool:
        return self.action in ("long", "short")

    def is_exit(self) -> bool:
        return self.action == "exit"

    def is_directional(self) -> bool:
        return self.action in ("long", "short")

    def describe(self) -> str:
        """
        Human-readable summary for telemetry and debugging.
        """
        return (
            f"{self.minion_name}: action={self.action}, "
            f"score={self.score:.3f}, confidence={self.confidence:.3f}, "
            f"symbol={self.symbol}, meta={self.meta}"
        )
