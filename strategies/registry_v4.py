# mikebot/strategies/registry_v4.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable

import pandas as pd

from mikebot.strategies.base_strategy_v4 import BaseStrategyV4, StrategyState
from mikebot.strategies.strategy_registry import StrategyRegistry, StrategyMetadata


@dataclass
class StrategyInstanceBundle:
    strategy: BaseStrategyV4
    metadata: StrategyMetadata
    state: StrategyState


class StrategyFactoryV4:
    """
    Factory that binds StrategyRegistry metadata + StrategyImpl classes
    into BaseStrategyV4-compatible instances with state.
    """

    def __init__(self, registry: StrategyRegistry) -> None:
        self._registry = registry

    def create(
        self,
        strategy_id: str,
        config: Dict[str, Any],
        state: StrategyState,
    ) -> Optional[StrategyInstanceBundle]:
        meta = self._registry.get_strategy(strategy_id)
        if not meta:
            return None

        logic_cls: Optional[Callable] = self._registry.get_logic(strategy_id)
        if logic_cls is None:
            return None

        impl = logic_cls(config)

        class WrappedStrategy(BaseStrategyV4):
            id = meta.id
            name = meta.name
            version = "1.0"
            category = meta.category.value

            def __init__(self, config: Dict[str, Any], state: StrategyState) -> None:
                super().__init__(config=config, state=state)
                self._impl = impl

            def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
                return self._impl.compute(df)

        strat = WrappedStrategy(config=config, state=state)
        return StrategyInstanceBundle(strategy=strat, metadata=meta, state=state)