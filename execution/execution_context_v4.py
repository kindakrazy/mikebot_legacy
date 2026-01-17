# mikebot/execution/execution_context_v4.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class BrokerAccountStateV4:
    """
    Snapshot of broker-level account state at a given moment.
    """
    balance: float
    equity: float
    margin_used: float
    margin_free: float
    currency: str
    leverage: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BrokerPositionV4:
    """
    Representation of an open position in the broker.
    """
    id: str
    symbol: str
    side: str       # "long" or "short"
    size: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BrokerOrderV4:
    """
    Representation of a pending order in the broker.
    """
    id: str
    symbol: str
    side: str       # "buy" or "sell"
    order_type: str # "market", "limit", "stop"
    size: float
    price: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketTickV4:
    """
    Minimal market tick or candle snapshot for execution.
    """
    symbol: str
    timestamp: Any
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    candle: Optional[pd.Series] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelDecisionV4:
    """
    Output of the model at execution time.
    """
    model_id: Optional[str]
    prediction: Any
    raw_output: Any = None
    features: Optional[Dict[str, Any]] = None


@dataclass
class StrategyDecisionV4:
    """
    Output of the strategy engine at execution time.
    """
    strategy_name: str
    signal: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionContextV4:
    """
    Canonical v4 execution context.

    This object is passed into the execution engine and contains:
    - market tick or candle
    - broker account state
    - open positions
    - pending orders
    - model decision
    - strategy decisions
    - free-form metadata
    """
    market: MarketTickV4
    account: BrokerAccountStateV4
    positions: List[BrokerPositionV4] = field(default_factory=list)
    orders: List[BrokerOrderV4] = field(default_factory=list)
    model_decision: Optional[ModelDecisionV4] = None
    strategy_decisions: List[StrategyDecisionV4] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)

    # --------------------------------------------------------------
    # Convenience helpers
    # --------------------------------------------------------------

    @property
    def symbol(self) -> str:
        return self.market.symbol

    def tag(self, key: str, value: Any) -> None:
        """
        Attach arbitrary metadata to the execution context.
        """
        self.extras[key] = value

    def get_position(self, pos_id: str) -> Optional[BrokerPositionV4]:
        """
        Retrieve a position by ID.
        """
        for p in self.positions:
            if p.id == pos_id:
                return p
        return None

    def get_order(self, order_id: str) -> Optional[BrokerOrderV4]:
        """
        Retrieve a pending order by ID.
        """
        for o in self.orders:
            if o.id == order_id:
                return o
        return None