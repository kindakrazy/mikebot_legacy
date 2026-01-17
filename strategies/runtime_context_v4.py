# mikebot/strategies/runtime_context_v4.py

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional, Mapping, Sequence, List

import pandas as pd


class StrategyMode(Enum):
    """
    High-level execution mode for strategies.

    - TRAINING: offline, batch data, labels available
    - BACKTEST: historical replay, no live orders
    - LIVE: real-time trading, live orders
    - SIMULATION: paper trading / sandbox
    """
    TRAINING = auto()
    BACKTEST = auto()
    LIVE = auto()
    SIMULATION = auto()


@dataclass
class MarketSnapshot:
    """Minimal market view for a strategy at a given decision point."""
    symbol: str
    timeframe: str
    current_bar: pd.Series
    history_window: pd.DataFrame = field(repr=False)
    multi_tf_features: Optional[pd.DataFrame] = field(default=None, repr=False)


@dataclass
class ModelSnapshot:
    """View of the model state relevant to a strategy decision."""
    model_id: Optional[str] = None
    model_type: Optional[str] = None
    latest_prediction: Optional[float] = None
    latest_features: Optional[Mapping[str, Any]] = None
    raw_output: Optional[Any] = None


@dataclass
class AccountSnapshot:
    """Lightweight account / risk view for strategies that care about sizing."""
    balance: Optional[float] = None
    equity: Optional[float] = None
    margin_used: Optional[float] = None
    margin_free: Optional[float] = None
    open_risk: Optional[float] = None
    currency: Optional[str] = None


@dataclass
class PositionSnapshot:
    """Minimal representation of an open position or order."""
    id: str
    symbol: str
    side: str  # "long" or "short"
    size: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyConfigView:
    """Strategy-specific configuration as seen at runtime."""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    raw_block: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyRuntimeContext:
    """
    Canonical v4 runtime context passed around strategy/execution layers.
    """
    mode: StrategyMode
    market: MarketSnapshot
    model: Optional[ModelSnapshot] = None
    account: Optional[AccountSnapshot] = None
    open_positions: List[PositionSnapshot] = field(default_factory=list)
    pending_orders: List[PositionSnapshot] = field(default_factory=list)
    strategy_config: Optional[StrategyConfigView] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def symbol(self) -> str:
        return self.market.symbol

    @property
    def timeframe(self) -> str:
        return self.market.timeframe

    @property
    def current_bar(self) -> pd.Series:
        return self.market.current_bar

    @property
    def history(self) -> pd.DataFrame:
        return self.market.history_window

    def get_param(self, key: str, default: Any = None) -> Any:
        if self.strategy_config is None:
            return default
        return self.strategy_config.params.get(key, default)

    def tag(self, key: str, value: Any) -> None:
        self.extras[key] = value


def build_market_snapshot(
    symbol: str,
    timeframe: str,
    candles: pd.DataFrame,
    multi_tf_features: Optional[pd.DataFrame] = None,
) -> MarketSnapshot:
    if candles is None or candles.empty:
        raise ValueError("build_market_snapshot: candles DataFrame is empty")

    current_bar = candles.iloc[-1]
    history_window = candles

    return MarketSnapshot(
        symbol=symbol,
        timeframe=timeframe,
        current_bar=current_bar,
        history_window=history_window,
        multi_tf_features=multi_tf_features,
    )


def build_model_snapshot(
    model_id: Optional[str],
    model_type: Optional[str],
    latest_prediction: Optional[float] = None,
    latest_features: Optional[Mapping[str, Any]] = None,
    raw_output: Optional[Any] = None,
) -> ModelSnapshot:
    return ModelSnapshot(
        model_id=model_id,
        model_type=model_type,
        latest_prediction=latest_prediction,
        latest_features=dict(latest_features) if latest_features is not None else None,
        raw_output=raw_output,
    )


def build_account_snapshot(
    balance: Optional[float] = None,
    equity: Optional[float] = None,
    margin_used: Optional[float] = None,
    margin_free: Optional[float] = None,
    open_risk: Optional[float] = None,
    currency: Optional[str] = None,
) -> AccountSnapshot:
    return AccountSnapshot(
        balance=balance,
        equity=equity,
        margin_used=margin_used,
        margin_free=margin_free,
        open_risk=open_risk,
        currency=currency,
    )


def build_strategy_config_view(
    name: str,
    params: Mapping[str, Any],
    raw_block: Optional[Mapping[str, Any]] = None,
) -> StrategyConfigView:
    return StrategyConfigView(
        name=name,
        params=dict(params),
        raw_block=dict(raw_block) if raw_block is not None else dict(params),
    )


def build_runtime_context(
    mode: StrategyMode,
    symbol: str,
    timeframe: str,
    candles: pd.DataFrame,
    *,
    multi_tf_features: Optional[pd.DataFrame] = None,
    model_snapshot: Optional[ModelSnapshot] = None,
    account_snapshot: Optional[AccountSnapshot] = None,
    open_positions: Optional[Sequence[PositionSnapshot]] = None,
    pending_orders: Optional[Sequence[PositionSnapshot]] = None,
    strategy_config: Optional[StrategyConfigView] = None,
    extras: Optional[Dict[str, Any]] = None,
) -> StrategyRuntimeContext:
    market = build_market_snapshot(
        symbol=symbol,
        timeframe=timeframe,
        candles=candles,
        multi_tf_features=multi_tf_features,
    )

    return StrategyRuntimeContext(
        mode=mode,
        market=market,
        model=model_snapshot,
        account=account_snapshot,
        open_positions=list(open_positions) if open_positions is not None else [],
        pending_orders=list(pending_orders) if pending_orders is not None else [],
        strategy_config=strategy_config,
        extras=dict(extras) if extras is not None else {},
    )