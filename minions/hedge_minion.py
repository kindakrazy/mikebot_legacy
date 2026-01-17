from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import logging
import numpy as np
import pandas as pd

from .minions_base import (
    Minion,
    MinionContext,
    MinionDecision,
    OrderRequest,
    OrderSide,
)
from .portfolio import PortfolioOptimizerConfig
from .max_lot_calc import MaxLotCalculator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HedgeMinionConfig:
    """
    Configuration for the hedge minion.

    Distilled from:
      - HighstrikeSignals/plugins/hedge_minion.py
      - HighstrikeSignals/modules/portfolio_optimizer.py
      - HighstrikeSignals/modules/risk/max_lot_calc.py
      - HighstrikeSignals/modules/survivability.py
    """
    enabled: bool = True

    # Symbol relationships: {"EURUSD": "DXY", "XAUUSD": "DXY"}
    hedge_pairs: Dict[str, str] = None
    correlation_lookback: int = 250

    # Risk parameters
    max_hedge_fraction: float = 0.5
    min_correlation: float = 0.3
    min_notional: float = 100.0

    # Portfolio optimizer alignment
    portfolio_config: Optional[PortfolioOptimizerConfig] = None

    # Max lot calculator config
    max_lot_config: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.hedge_pairs is None:
            self.hedge_pairs = {}
        if self.max_lot_config is None:
            self.max_lot_config = {}


# ---------------------------------------------------------------------------
# HedgeMinion
# ---------------------------------------------------------------------------

class HedgeMinion(Minion):
    """
    Hedging and risk‑balancing minion.

    Modernized to:
      - return MinionDecision(action/score/confidence/meta)
      - not emit raw orders directly
      - expose hedge orders via meta for PortfolioOptimizer to consume
    """

    name = "hedge_minion"

    def __init__(self, config: HedgeMinionConfig) -> None:
        self.config = config
        self._max_lot_calc = MaxLotCalculator(config.max_lot_config or {})

    # ----------------------------------------------------------------------
    # Minion API
    # ----------------------------------------------------------------------

    def decide(self, ctx: MinionContext) -> MinionDecision:
        if not self.config.enabled:
            return MinionDecision(
                minion_name=self.name,
                action="hold",
                score=0.0,
                confidence=0.0,
                symbol=None,
                meta={"reason": "disabled"},
            )

        try:
            exposures = self._compute_exposures(ctx.open_positions)
            hedge_orders = self._build_hedge_orders(ctx, exposures)

            total_hedge_notional = sum(
                o.lot_size * (self._latest_price(ctx, o.symbol) or 0.0)
                for o in hedge_orders
            )
            confidence = 1.0 if hedge_orders and total_hedge_notional > 0 else 0.0

            return MinionDecision(
                minion_name=self.name,
                action="hold",
                score=0.0,
                confidence=float(confidence),
                symbol=None,
                meta={
                    "exposures": exposures,
                    "num_hedge_orders": len(hedge_orders),
                    "total_hedge_notional": total_hedge_notional,
                    "hedge_orders": [o.to_dict() for o in hedge_orders],
                },
            )

        except Exception as exc:
            logger.exception("HedgeMinion.decide failed: %s", exc)
            return MinionDecision(
                minion_name=self.name,
                action="hold",
                score=0.0,
                confidence=0.0,
                symbol=None,
                meta={"error": str(exc)},
            )

    # ----------------------------------------------------------------------
    # Exposure computation
    # ----------------------------------------------------------------------

    def _compute_exposures(
        self,
        open_positions: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Compute per‑symbol notional exposure.
        Positive = net long, negative = net short.
        """
        exposures: Dict[str, float] = {}

        for pos in open_positions:
            symbol = pos.get("symbol")
            lots = float(pos.get("lots", 0.0))
            side = pos.get("side", "BUY").upper()
            price = float(pos.get("price", 0.0))

            if not symbol or price <= 0:
                continue

            direction = 1.0 if side in ("BUY", "LONG") else -1.0
            notional = lots * price * direction

            exposures[symbol] = exposures.get(symbol, 0.0) + notional

        return exposures

    # ----------------------------------------------------------------------
    # Hedge construction
    # ----------------------------------------------------------------------

    def _build_hedge_orders(
        self,
        ctx: MinionContext,
        exposures: Dict[str, float],
    ) -> List[OrderRequest]:
        orders: List[OrderRequest] = []

        for base_symbol, exposure in exposures.items():
            if abs(exposure) < self.config.min_notional:
                continue

            hedge_symbol = self.config.hedge_pairs.get(base_symbol)
            if not hedge_symbol:
                continue

            base_price = self._latest_price(ctx, base_symbol)
            hedge_price = self._latest_price(ctx, hedge_symbol)
            if base_price is None or hedge_price is None:
                continue

            corr = self._estimate_correlation(ctx, base_symbol, hedge_symbol)
            if corr is None or abs(corr) < self.config.min_correlation:
                continue

            hedge_notional = -np.sign(exposure) * abs(exposure) * self.config.max_hedge_fraction
            hedge_lots = hedge_notional / hedge_price if hedge_price > 0 else 0.0

            if abs(hedge_lots * hedge_price) < self.config.min_notional:
                continue

            # Respect max lot constraints
            max_lot = self._max_lot_calc.compute(
                symbol=hedge_symbol,
                account_state=ctx.account_state,
                open_positions=ctx.open_positions,
            )
            hedge_lots = float(np.clip(hedge_lots, -max_lot, max_lot))
            if abs(hedge_lots) <= 0:
                continue

            side = OrderSide.BUY if hedge_lots > 0 else OrderSide.SELL

            order = OrderRequest(
                symbol=hedge_symbol,
                side=side,
                lot_size=abs(hedge_lots),
                price=None,
                stop_loss=None,
                take_profit=None,
                comment=f"hedge_minion hedge for {base_symbol} (corr={corr:.2f})",
            )
            orders.append(order)

        return orders

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    def _latest_price(self, ctx: MinionContext, symbol: str) -> Optional[float]:
        feat = ctx.features_by_symbol.get(symbol)
        if feat is None:
            return None

        df = feat.get("features")
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None

        if "close" in df.columns:
            return float(df["close"].iloc[-1])

        for col in ("price", "mid", "last"):
            if col in df.columns:
                return float(df[col].iloc[-1])

        return None

    def _estimate_correlation(
        self,
        ctx: MinionContext,
        base_symbol: str,
        hedge_symbol: str,
    ) -> Optional[float]:
        """
        Estimate correlation between base and hedge symbol returns.
        Uses ret_1 if available; falls back to pct_change().
        """
        base_feat = ctx.features_by_symbol.get(base_symbol)
        hedge_feat = ctx.features_by_symbol.get(hedge_symbol)
        if base_feat is None or hedge_feat is None:
            return None

        base_df = base_feat.get("features")
        hedge_df = hedge_feat.get("features")
        if not isinstance(base_df, pd.DataFrame) or not isinstance(hedge_df, pd.DataFrame):
            return None
        if base_df.empty or hedge_df.empty:
            return None

        base = base_df.tail(self.config.correlation_lookback)
        hedge = hedge_df.tail(self.config.correlation_lookback)

        # Prefer ret_1 if present
        if "ret_1" in base.columns and "ret_1" in hedge.columns:
            base_ret = base["ret_1"].dropna()
            hedge_ret = hedge["ret_1"].dropna()
        else:
            # Fallback to raw close returns
            if "close" not in base.columns or "close" not in hedge.columns:
                return None
            merged = pd.DataFrame(
                {
                    "base": base["close"],
                    "hedge": hedge["close"],
                }
            ).dropna()
            if len(merged) < 20:
                return None
            base_ret = merged["base"].pct_change().dropna()
            hedge_ret = merged["hedge"].pct_change().dropna()

        if len(base_ret) < 10 or len(hedge_ret) < 10:
            return None

        merged2 = pd.concat([base_ret, hedge_ret], axis=1, join="inner").dropna()
        if len(merged2) < 10:
            return None

        corr = float(merged2.corr().iloc[0, 1])
        return corr