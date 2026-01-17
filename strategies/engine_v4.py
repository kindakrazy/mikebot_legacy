# mikebot/strategies/engine_v4.py

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Mapping

import pandas as pd

from mikebot.live.orchestrator.config import LiveConfig
from mikebot.strategies.base_strategy_v4 import StrategyState
from mikebot.strategies.strategy_registry import StrategyRegistry
from mikebot.strategies.registry_v4 import StrategyFactoryV4, StrategyInstanceBundle
from mikebot.strategies.runtime_context_v4 import StrategyRuntimeContext, StrategyMode

logger = logging.getLogger(__name__)


@dataclass
class StrategyEngineResult:
    strategy_id: str
    name: str
    features: pd.DataFrame
    state: StrategyState = field(repr=False)


class StrategyEngineV4:
    """
    Full V4 strategy engine.

    - Multi-strategy
    - Stateful
    - LiveConfig-aware
    - Runtime-context aware
    """

    def __init__(
        self,
        live_config: LiveConfig,
        *,
        registry: Optional[StrategyRegistry] = None,
        toggles: Optional[Mapping[str, bool]] = None,
    ) -> None:
        self.live_config = live_config
        self.registry = registry or StrategyRegistry()
        self.factory = StrategyFactoryV4(self.registry)
        self._toggles: Mapping[str, bool] = toggles or {}
        self._states: Dict[str, StrategyState] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_enabled(self, strategy_id: str) -> bool:
        """Check if a strategy is enabled by toggles (default: enabled)."""
        if not self._toggles:
            return True
        return self._toggles.get(strategy_id, True)

    def _pattern_name_for(self, strategy_id: str) -> Optional[str]:
        """Map strategy_id -> pattern name used in LiveConfig.strategy_patterns."""
        mapping: Dict[str, str] = {
            "strat_bear_flag_v1": "BearFlag",
            "strat_bull_flag_v1": "BullFlag",
        }
        return mapping.get(strategy_id)

    def _config_for(self, strategy_id: str) -> Dict[str, Any]:
        pattern = self._pattern_name_for(strategy_id)
        if not pattern:
            return {}
        return self.live_config.strategy_params(pattern)

    def _state_for(self, strategy_id: str) -> StrategyState:
        if strategy_id not in self._states:
            self._states[strategy_id] = StrategyState()
        return self._states[strategy_id]

    def _build_instance(self, strategy_id: str) -> Optional[StrategyInstanceBundle]:
        cfg = self._config_for(strategy_id)
        state = self._state_for(strategy_id)
        return self.factory.create(strategy_id=strategy_id, config=cfg, state=state)

    # ------------------------------------------------------------------
    # Batch / training-time API
    # ------------------------------------------------------------------

    def run_batch(
        self,
        candles: pd.DataFrame,
        strategy_ids: Optional[List[str]] = None,
    ) -> Dict[str, StrategyEngineResult]:
        """
        Run multiple strategies over a candle DataFrame (training / backtest).
        """
        if candles is None or candles.empty:
            raise ValueError("StrategyEngineV4.run_batch: candles DataFrame is empty")

        if strategy_ids is None:
            strategy_ids = self.registry.get_all_ids()

        results: Dict[str, StrategyEngineResult] = {}

        for sid in strategy_ids:
            if not self._is_enabled(sid):
                continue

            bundle = self._build_instance(sid)
            if bundle is None:
                continue

            try:
                feats = bundle.strategy.run(candles)
            except Exception as exc:
                logger.exception("StrategyEngineV4: Strategy '%s' failed in run_batch: %s", sid, exc)
                continue

            if feats is None or feats.empty:
                continue

            results[sid] = StrategyEngineResult(
                strategy_id=sid,
                name=bundle.metadata.name,
                features=feats,
                state=bundle.state,
            )

        return results

    def as_feature_mapping(
        self,
        batch: Dict[str, StrategyEngineResult],
    ) -> Dict[str, pd.DataFrame]:
        """
        Convert engine results into namespaced feature frames.
        """
        feature_map: Dict[str, pd.DataFrame] = {}
        for sid, res in batch.items():
            df = res.features
            if df is None or df.empty:
                continue
            ns_df = df.copy()
            ns_df.columns = [f"{res.name}__{c}" for c in ns_df.columns]
            feature_map[sid] = ns_df
        return feature_map

    # ------------------------------------------------------------------
    # Execution-time API (runtime context)
    # ------------------------------------------------------------------

    def run_for_context(
        self,
        ctx: StrategyRuntimeContext,
        strategy_ids: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Run strategies for a single runtime context.

        Returns:
            {strategy_id: latest_signal_value}
        """
        candles = ctx.history
        if candles is None or not isinstance(candles, pd.DataFrame) or candles.empty:
            raise ValueError("StrategyEngineV4.run_for_context: ctx.history must be a non-empty DataFrame")

        if strategy_ids is None:
            strategy_ids = self.registry.get_all_ids()

        outputs: Dict[str, float] = {}

        for sid in strategy_ids:
            if not self._is_enabled(sid):
                continue

            bundle = self._build_instance(sid)
            if bundle is None:
                continue

            try:
                feats = bundle.strategy.run(candles)
            except Exception as exc:
                logger.exception("StrategyEngineV4: Strategy '%s' failed in run_for_context: %s", sid, exc)
                continue

            if feats is None or feats.empty or "signal" not in feats.columns:
                continue

            outputs[sid] = float(feats["signal"].iloc[-1])

        return outputs

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def snapshot_state(self) -> Dict[str, Any]:
        return {
            sid: {
                "last_signal": st.last_signal,
                "extras": dict(st.extras),
            }
            for sid, st in self._states.items()
        }
# --- Global accessor for StrategyEngineV4 ---

_strategy_engine_global = None


def set_global_strategy_engine(engine) -> None:
    """
    Register the process-wide StrategyEngineV4 instance.

    Call this once during startup, after constructing the engine.
    """
    global _strategy_engine_global
    _strategy_engine_global = engine


def get_global_strategy_engine():
    """
    Return the process-wide StrategyEngineV4 instance, or None if not set.
    """
    return _strategy_engine_global