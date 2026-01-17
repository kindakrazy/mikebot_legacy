# mikebot/strategies/base_strategy_v4.py

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

import pandas as pd


@dataclass
class StrategyState:
    """Per-strategy mutable state."""
    last_signal: float = 0.0
    extras: Dict[str, Any] = field(default_factory=dict)


class BaseStrategyV4(ABC):
    """
    Canonical V4 strategy interface.

    Concrete strategies wrap underlying implementations (e.g. BearFlag/BullFlag)
    and expose a unified run() API with state.
    """

    id: str
    name: str
    version: str
    category: str

    def __init__(self, config: Dict[str, Any], state: Optional[StrategyState] = None) -> None:
        self.config = config
        self.state = state or StrategyState()

    @abstractmethod
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute ML-friendly features for this strategy.
        Must return a DataFrame indexed like df, with at least a 'signal' column.
        """
        raise NotImplementedError

    def update_state(self, features: pd.DataFrame) -> None:
        """Update internal state from computed features."""
        if "signal" in features.columns and not features.empty:
            self.state.last_signal = float(features["signal"].iloc[-1])

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full strategy step: compute features, update state, return features."""
        features = self.compute_features(df)
        self.update_state(features)
        return features