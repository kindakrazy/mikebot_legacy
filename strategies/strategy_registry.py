# mikebot/strategies/strategy_registry.py

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Callable

from mikebot.strategies.strategies.bear_flag import StrategyImpl as BearFlagStrategy
from mikebot.strategies.strategies.bull_flag import StrategyImpl as BullFlagStrategy

logger = logging.getLogger(__name__)


class StrategyCategory(Enum):
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    VOLATILITY_BREAKOUT = "volatility_breakout"
    SCALPING = "scalping"


@dataclass
class StrategyMetadata:
    """Canonical definition of a Mikebot Strategy."""
    id: str
    name: str
    category: StrategyCategory
    primary_indicator: str
    description: str
    params: Dict[str, Any] = field(default_factory=dict)


class StrategyRegistry:
    """
    Central repository for all trading logic definitions.

    Maps Strategy IDs to metadata and logic implementations.
    """

    def __init__(self) -> None:
        self._strategies: Dict[str, StrategyMetadata] = {}
        self._logic_map: Dict[str, Callable] = {}
        self._register_defaults()

    def register(self, meta: StrategyMetadata, logic_func: Optional[Callable] = None) -> None:
        """Adds a new strategy to the system."""
        self._strategies[meta.id] = meta
        if logic_func:
            self._logic_map[meta.id] = logic_func
        logger.info("StrategyRegistry: Registered '%s' [%s]", meta.name, meta.id)

    def get_strategy(self, strategy_id: str) -> Optional[StrategyMetadata]:
        return self._strategies.get(strategy_id)

    def get_logic(self, strategy_id: str) -> Optional[Callable]:
        return self._logic_map.get(strategy_id)

    def list_by_category(self, category: StrategyCategory) -> List[StrategyMetadata]:
        return [s for s in self._strategies.values() if s.category == category]

    def get_all_ids(self) -> List[str]:
        return list(self._strategies.keys())

    def _register_defaults(self) -> None:
        """Wires up the core Mikebot strategy baselines."""

        # 1. Volatility Breakout (placeholder metadata only)
        self.register(
            StrategyMetadata(
                id="strat_vol_breakout_v1",
                name="Volatility Breakout",
                category=StrategyCategory.VOLATILITY_BREAKOUT,
                primary_indicator="ATR_Keltner",
                description="Trades moves outside of volatility-adjusted bands.",
                params={"atr_period": 14, "multiplier": 2.0},
            )
        )

        # 2. Mean Reversion (placeholder metadata only)
        self.register(
            StrategyMetadata(
                id="strat_rsi_reversion_v1",
                name="RSI Mean Reversion",
                category=StrategyCategory.MEAN_REVERSION,
                primary_indicator="RSI",
                description="Enters trades on extreme RSI exhaustion points.",
                params={"rsi_period": 14, "overbought": 70, "oversold": 30},
            )
        )

        # 3. Trend Following (EMA Cross, placeholder)
        self.register(
            StrategyMetadata(
                id="strat_ema_cross_v1",
                name="EMA Golden Cross",
                category=StrategyCategory.TREND_FOLLOWING,
                primary_indicator="EMA",
                description="Standard trend following using dual EMA crossovers.",
                params={"fast": 50, "slow": 200},
            )
        )

        # 4. Bear Flag (wired to StrategyImpl)
        self.register(
            StrategyMetadata(
                id="strat_bear_flag_v1",
                name="BearFlag",
                category=StrategyCategory.TREND_FOLLOWING,
                primary_indicator="Price Action",
                description="Bearish continuation pattern with controlled pullback.",
                params={},
            ),
            logic_func=BearFlagStrategy,
        )

        # 5. Bull Flag (wired to StrategyImpl)
        self.register(
            StrategyMetadata(
                id="strat_bull_flag_v1",
                name="BullFlag",
                category=StrategyCategory.TREND_FOLLOWING,
                primary_indicator="Price Action",
                description="Bullish continuation pattern with controlled pullback.",
                params={},
            ),
            logic_func=BullFlagStrategy,
        )

    def get_strategy_context(self, strategy_id: str) -> Dict[str, Any]:
        """Returns a dict representation for use in ExperienceStore tagging."""
        strat = self.get_strategy(strategy_id)
        if not strat:
            return {"strategy_id": "unknown"}
        return {
            "strategy_id": strat.id,
            "category": strat.category.value,
            "params": strat.params,
        }