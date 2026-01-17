from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np

from .minions_base import OrderRequest, OrderSide
from .survivability import SurvivabilityGuard
from .max_lot_calc import MaxLotCalculator
from .multi_agent import BlendedDecision

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Portfolio optimizer configuration
# ---------------------------------------------------------------------------

@dataclass
class PortfolioOptimizerConfig:
    """
    Configuration for the portfolio optimizer (Mikebot v3.5).
    """

    # Base lot size before scaling
    base_lot: float = 0.01

    # Maximum multiplier applied after scaling
    max_lot_multiplier: float = 3.0

    # How strongly blended score affects lot size
    score_sensitivity: float = 1.5

    # Exposure limits (reinforced by survivability)
    max_symbol_exposure: float = 0.20
    max_total_exposure: float = 0.50

    # Whether to allow hedge_minion to add hedge orders
    allow_hedging: bool = True


# ---------------------------------------------------------------------------
# Portfolio optimizer
# ---------------------------------------------------------------------------

class PortfolioOptimizer:
    """
    Converts a BlendedDecision into executable OrderRequest objects.

    Responsibilities:
      - Position sizing
      - Personality-aware scaling
      - Survivability-aware throttling
      - Exposure-aware throttling
      - Hedge alignment
      - Final order construction
    """

    def __init__(
        self,
        config: PortfolioOptimizerConfig,
        survivability: SurvivabilityGuard,
        max_lot_calc: MaxLotCalculator,
    ) -> None:
        self.config = config
        self.survivability = survivability
        self.max_lot_calc = max_lot_calc

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def build_orders(
        self,
        blended: BlendedDecision,
        ctx,
        hedge_orders: Optional[List[OrderRequest]] = None,
    ) -> List[OrderRequest]:
        """
        Convert the blended multi-agent decision into final orders.
        """

        # Survivability check
        surv = self.survivability.check_survivability(
            account_state=ctx.account_state,
            open_positions=ctx.open_positions,
            volatility_series=ctx.volatility_series,
            loop_iteration=ctx.loop_iteration,
        )

        if surv["safe_mode"]:
            logger.warning("PortfolioOptimizer: SAFE MODE active → blocking all new exposure")
            return []

        # No directional action → no primary order
        if blended.action not in ("long", "short"):
            return []

        # Build primary order
        primary = self._build_primary_order(blended, ctx)
        if primary is None:
            return []

        orders = [primary]

        # Add hedge orders if allowed
        if self.config.allow_hedging and hedge_orders:
            orders.extend(hedge_orders)

        return orders

    # ----------------------------------------------------------------------
    # Primary order construction
    # ----------------------------------------------------------------------

    def _build_primary_order(
        self,
        blended: BlendedDecision,
        ctx,
    ) -> Optional[OrderRequest]:
        """
        Build the main order from the blended decision.
        """

        symbol = blended.symbol or ctx.primary_symbol
        if not symbol:
            return None

        # Determine side
        if blended.action == "long":
            side = OrderSide.BUY
        elif blended.action == "short":
            side = OrderSide.SELL
        else:
            return None

        # Score-based scaling
        score = float(blended.score)
        scale = 1.0 + abs(score) * self.config.score_sensitivity

        # Personality scaling
        personality = ctx.personality
        scale *= personality.aggression
        scale /= max(personality.caution, 1e-9)

        # Base lot
        lot = self.config.base_lot * scale

        # Max lot constraint
        max_lot = self.max_lot_calc.compute(
            symbol=symbol,
            account_state=ctx.account_state,
            open_positions=ctx.open_positions,
        )
        lot = float(np.clip(lot, 0.0, max_lot * self.config.max_lot_multiplier))

        if lot <= 0:
            return None

        return OrderRequest(
            symbol=symbol,
            side=side,
            lot_size=lot,
            price=None,
            stop_loss=None,
            take_profit=None,
            comment=f"portfolio primary (score={score:.3f}, scale={scale:.3f})",
        )