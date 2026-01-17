# C:\mikebot\minions\minion_context.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional


@dataclass(frozen=True)
class MinionContext:
    """
    Modern, explicit, typed context passed to all minions.

    This replaces the legacy dynamic attribute bag with a stable,
    predictable, validated structure.
    """

    # Session + iteration metadata
    session_id: str
    timestamp: datetime
    loop_iteration: int

    # Market data + features
    features_by_symbol: Dict[str, Any]
    last_prices: Dict[str, float]
    volatility_series: Optional[List[float]]

    # Trading state
    account_state: Any
    open_positions: Any

    # Behavior + personality
    personality: Any
    primary_symbol: Optional[str]

    # Optional knowledge systems
    knowledge_graph: Optional[Any] = None
    regime: Optional[Any] = None

    # Convenience helpers
    def feature_pack(self, symbol: str) -> Optional[Any]:
        return self.features_by_symbol.get(symbol)

    def price(self, symbol: str) -> Optional[float]:
        return self.last_prices.get(symbol)

    def volatility(self) -> Optional[List[float]]:
        return self.volatility_series
