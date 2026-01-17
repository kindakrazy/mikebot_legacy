# C:\mikebot\mikebot\features\feature_interface.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class FeatureState:
    """
    Mutable state passed between feature computations.
    """
    values: Dict[str, Any]


class Feature:
    """
    Base interface for v4 streaming features.
    """

    def compute(self, candle, state: FeatureState) -> FeatureState:
        """
        Compute feature values for a single candle.

        Implementations must:
        - read from `candle`
        - update `state.values`
        - return the updated FeatureState
        """
        raise NotImplementedError