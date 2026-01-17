# mikebot/execution/execution_engine_v4.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, List

import pandas as pd

from mikebot.execution.execution_context_v4 import (
    ExecutionContextV4,
    BrokerAccountStateV4,
    BrokerPositionV4,
    BrokerOrderV4,
    MarketTickV4,
    ModelDecisionV4,
    StrategyDecisionV4,
)
from mikebot.execution.order_router_v4 import OrderRouterV4, OrderResultV4
from mikebot.strategies.engine_v4 import StrategyEngineV4
from mikebot.strategies.runtime_context_v4 import (
    StrategyRuntimeContext,
    StrategyMode,
    MarketSnapshot,
    ModelSnapshot,
    AccountSnapshot,
)


@dataclass
class ExecutionDecisionV4:
    """
    High-level decision produced by the execution engine.

    This is a normalized representation of what the engine decided to do
    at a given tick:
    - no trade
    - open/close/modify
    - with optional metadata for logging.
    """
    action: str  # "none", "open", "close", "modify"
    symbol: str
    side: Optional[str] = None
    size: Optional[float] = None
    price: Optional[float] = None
    metadata: Dict[str, Any] = None


class ExecutionEngineV4:
    """
    Canonical v4 execution engine.

    Responsibilities:
    - Consume live ticks or candles.
    - Build a StrategyRuntimeContext for the v4 strategy engine.
    - Run strategies to obtain signals.
    - Run the model (optional) to obtain predictions.
    - Combine strategy + model into an ExecutionDecisionV4.
    - Route orders via OrderRouterV4.
    - Produce an updated ExecutionContextV4 for logging and downstream use.

    This engine is intentionally generic and broker-agnostic.
    """

    def __init__(
        self,
        *,
        strategy_engine: StrategyEngineV4,
        order_router: OrderRouterV4,
        model: Optional[Any] = None,
        symbol: str,
        timeframe: str,
        base_size: float = 1.0,
    ) -> None:
        self.strategy_engine = strategy_engine
        self.order_router = order_router
        self.model = model
        self.symbol = symbol
        self.timeframe = timeframe
        self.base_size = base_size

        # Rolling candle history for context
        self._history: List[pd.Series] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_candle(
        self,
        candle: pd.Series,
        account_state: BrokerAccountStateV4,
        positions: Sequence[BrokerPositionV4],
        orders: Sequence[BrokerOrderV4],
    ) -> ExecutionContextV4:
        """
        Main entrypoint: called once per new candle.

        Returns an updated ExecutionContextV4 after:
        - building runtime context
        - running strategies
        - running model (optional)
        - making a trade decision
        - routing orders (if any)
        """
        self._history.append(candle)
        history_df = pd.DataFrame(self._history)
        history_df.index = range(len(history_df))

        # Build market snapshot for strategy runtime
        market_snapshot = MarketSnapshot(
            symbol=self.symbol,
            timeframe=self.timeframe,
            current_bar=candle,
            history_window=history_df,
        )

        # Account snapshot for strategy runtime
        account_snapshot = AccountSnapshot(
            balance=account_state.balance,
            equity=account_state.equity,
            margin_used=account_state.margin_used,
            margin_free=account_state.margin_free,
            open_risk=None,
            currency=account_state.currency,
        )

        # Model snapshot (optional, filled after prediction)
        model_snapshot = None

        # Build strategy runtime context
        runtime_ctx = StrategyRuntimeContext(
            mode=StrategyMode.LIVE,
            market=market_snapshot,
            model=model_snapshot,
            account=account_snapshot,
            open_positions=[],
            pending_orders=[],
            strategy_config=None,
            extras={},
        )

        # Run strategies
        strategy_outputs = self.strategy_engine.run_single(runtime_ctx)
        strategy_decisions: List[StrategyDecisionV4] = []
        for name, out in strategy_outputs.items():
            strategy_decisions.append(
                StrategyDecisionV4(
                    strategy_name=name,
                    signal=out,
                    metadata={},
                )
            )

        # Run model (optional)
        model_decision = None
        if self.model is not None:
            features = self.strategy_engine.extract_features_for_model(runtime_ctx, strategy_outputs)
            pred = self.model.predict(features.values.reshape(1, -1))[0]
            model_decision = ModelDecisionV4(
                model_id=getattr(self.model, "model_id", None),
                prediction=pred,
                raw_output=pred,
                features=features.to_dict(),
            )

        # Combine into execution decision
        decision = self._make_decision(
            candle=candle,
            strategy_decisions=strategy_decisions,
            model_decision=model_decision,
        )

        # Route orders if needed
        if decision.action == "open" and decision.side and decision.size:
            self._route_open(decision)
        elif decision.action == "close":
            # For simplicity, close all positions on this symbol
            self._route_close_all(self.symbol)

        # Build execution context
        market_tick = MarketTickV4(
            symbol=self.symbol,
            timestamp=candle.name,
            bid=None,
            ask=None,
            last=float(candle["close"]),
            candle=candle,
            raw={},
        )

        exec_ctx = ExecutionContextV4(
            market=market_tick,
            account=account_state,
            positions=list(positions),
            orders=list(orders),
            model_decision=model_decision,
            strategy_decisions=strategy_decisions,
            extras={"decision": decision},
        )

        return exec_ctx

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def _make_decision(
        self,
        *,
        candle: pd.Series,
        strategy_decisions: Sequence[StrategyDecisionV4],
        model_decision: Optional[ModelDecisionV4],
    ) -> ExecutionDecisionV4:
        """
        Combine strategy and model outputs into a single execution decision.

        This is intentionally simple and can be replaced with a more
        sophisticated policy later.
        """
        # Example policy:
        # - If any strategy emits "long" and model prediction > 0, open long.
        # - If any strategy emits "short" and model prediction < 0, open short.
        # - Otherwise, no trade.

        side = None
        for sd in strategy_decisions:
            if sd.signal == "long":
                side = "buy"
                break
            if sd.signal == "short":
                side = "sell"
                break

        if side is None:
            return ExecutionDecisionV4(
                action="none",
                symbol=self.symbol,
                metadata={"reason": "no_strategy_signal"},
            )

        if model_decision is not None:
            pred = model_decision.prediction
            if side == "buy" and pred <= 0:
                return ExecutionDecisionV4(
                    action="none",
                    symbol=self.symbol,
                    metadata={"reason": "model_disagrees_long", "prediction": pred},
                )
            if side == "sell" and pred >= 0:
                return ExecutionDecisionV4(
                    action="none",
                    symbol=self.symbol,
                    metadata={"reason": "model_disagrees_short", "prediction": pred},
                )

        return ExecutionDecisionV4(
            action="open",
            symbol=self.symbol,
            side=side,
            size=self.base_size,
            price=float(candle["close"]),
            metadata={"source": "simple_policy"},
        )

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def _route_open(self, decision: ExecutionDecisionV4) -> OrderResultV4:
        """
        Route an open-position decision as a market order.
        """
        return self.order_router.send_market_order(
            symbol=decision.symbol,
            side=decision.side or "buy",
            size=decision.size or self.base_size,
            metadata=decision.metadata or {},
        )

    def _route_close_all(self, symbol: str) -> None:
        """
        Close all positions for a symbol.

        This is a placeholder policy; a more advanced engine would
        track positions and close selectively.
        """
        # In a real implementation, we'd query positions and close them.
        # Here we just expose the hook.
        pass