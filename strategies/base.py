# mikebot/strategies/base.py

from __future__ import annotations

from typing import Any, Mapping, Protocol, Union, runtime_checkable

import pandas as pd


# ---------------------------------------------------------------------------
# Legacy contract (v1) – kept for backward compatibility
# ---------------------------------------------------------------------------

@runtime_checkable
class LegacyStrategy(Protocol):
    """
    Original Highstrike / Mikebot strategy contract.

    - Stateless
    - Single static compute()
    - Returns a 1D pd.Series of signals aligned to df.index
    - Signals are typically in { -1.0, 0.0, 1.0 }
    """

    name: str

    @staticmethod
    def compute(df: pd.DataFrame) -> pd.Series:
        """
        Must return a pd.Series aligned to df.index
        containing float signals (0.0, 1.0, or -1.0).
        """
        ...


# ---------------------------------------------------------------------------
# Modern contract (v2+) – used by BearFlag, BullFlag, etc.
# ---------------------------------------------------------------------------

@runtime_checkable
class ModernStrategy(Protocol):
    """
    Modern, ML‑friendly strategy contract.

    Implementations in mikebot/strategies/strategies (e.g. BearFlag, BullFlag)
    already follow this shape:

        class StrategyImpl:
            name = "BearFlag"
            version = "1.0"
            parameters = ["flagpole_min_atr", "pullback_min_pct", ...]

            def __init__(self, config: dict):
                self.cfg = config

            def compute(self, df: pd.DataFrame) -> pd.DataFrame:
                ...

    Key expectations:

    - __init__(config): receives a dict‑like config block.
    - compute(df): returns a DataFrame aligned to df.index OR with a
      'timestamp' column that can be normalized by the registry.
    - Output MUST contain at least a 'signal' column (float), where:
        * positive values bias long
        * negative values bias short
        * 0.0 means no opinion / neutral
    - Additional columns are treated as ML features (e.g. flagpole_strength).
    """

    # Human‑readable name (used in UI, toggles, configs)
    name: str

    # Optional semantic version of the strategy logic
    version: str

    # Parameter names used by UI / optimizer (if present)
    parameters: list[str]

    def __init__(self, config: Mapping[str, Any]) -> None:
        """
        Implementations receive a config mapping (typically from
        StrategyConfigLoader). They may store it as self.cfg or similar.
        """
        ...

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute strategy outputs for the given candle DataFrame.

        Requirements:
        - df contains at least: ['open', 'high', 'low', 'close', 'volume']
        - Returned DataFrame:
            * Either indexed identically to df.index, OR
            * Contains a 'timestamp' column that can be converted to a
              timezone‑aware datetime index by the registry.

        Columns:
        - 'signal' (required): float, continuous or discrete signal.
        - Any additional columns are treated as ML‑friendly features and
          will be namespaced when merged into the feature matrix.
        """
        ...


# ---------------------------------------------------------------------------
# Unified alias used throughout the codebase
# ---------------------------------------------------------------------------

Strategy = Union[LegacyStrategy, ModernStrategy]
"""
Unified strategy type used by the registry and callers.

Both of these are accepted:

- LegacyStrategy:
    class MyStrat:
        name = "MyStrat"

        @staticmethod
        def compute(df: pd.DataFrame) -> pd.Series:
            ...

- ModernStrategy:
    class StrategyImpl:
        name = "BearFlag"
        version = "1.0"
        parameters = [...]

        def __init__(self, config: dict):
            self.cfg = config

        def compute(self, df: pd.DataFrame) -> pd.DataFrame:
            ...

The registry and feature builder are written to tolerate both shapes
without breaking existing behavior.
"""