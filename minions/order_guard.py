# C:\mikebot\minions\order_guard.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .minions_base import OrderRequest
from .survivability import SurvivabilityGuard

logger = logging.getLogger(__name__)


@dataclass
class OrderGuardConfig:
    """
    Configuration for the order guard.

    This is the mikebot v2 distillation of:
      - HighstrikeSignals/modules/order_guard.py
      - HighstrikeSignals/modules/guardrails_check.py
      - HighstrikeSignals/config/guardrails.yml
      - HighstrikeSignals/config/schema/guardrails_schema.json
    """

    # Per-order constraints
    min_lot: float = 0.01
    max_lot: float = 10.0

    # Price sanity checks (relative to last price)
    max_slippage_pct: float = 0.02  # 2%

    # Stop-loss / take-profit constraints (in pips or relative terms)
    min_stop_distance_pct: float = 0.002  # 0.2%
    max_stop_distance_pct: float = 0.10   # 10%

    # Global toggles
    enforce_lot_limits: bool = True
    enforce_price_sanity: bool = True
    enforce_stop_sanity: bool = True


class OrderGuard:
    """
    Pre-trade order guard.

    Responsibilities:
      - Enforce basic per-order constraints (lot size, price sanity, SL/TP sanity)
      - Integrate with survivability guard for higher-level risk constraints
      - Provide a clean "filter" API for the order_router
    """

    def __init__(
        self,
        config: OrderGuardConfig,
        survivability: SurvivabilityGuard,
    ) -> None:
        self.config = config
        self.survivability = survivability

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def filter_orders(
        self,
        orders: List[OrderRequest],
        account_state: Dict[str, Any],
        open_positions: List[Dict[str, Any]],
        last_prices: Dict[str, float],
        loop_iteration: int,
        volatility_series,
    ) -> List[OrderRequest]:
        """
        Apply guardrails to a list of proposed orders.

        Returns only those orders that pass all checks.
        """
        if not orders:
            return []

        surv_state = self.survivability.check_survivability(
            account_state=account_state,
            open_positions=open_positions,
            volatility_series=volatility_series,
            loop_iteration=loop_iteration,
        )

        if surv_state.get("safe_mode"):
            logger.warning("OrderGuard: SAFE MODE active → blocking all new orders")
            return []

        filtered: List[OrderRequest] = []
        for o in orders:
            if self._check_order(o, account_state, last_prices):
                filtered.append(o)

        return filtered

    # ----------------------------------------------------------------------
    # Individual order checks
    # ----------------------------------------------------------------------

    def _check_order(
        self,
        order: OrderRequest,
        account_state: Dict[str, Any],
        last_prices: Dict[str, float],
    ) -> bool:
        """
        Check a single order against all configured constraints.
        """
        if self.config.enforce_lot_limits and not self._check_lot(order):
            logger.warning("OrderGuard: lot size violation for %s", order.symbol)
            return False

        if self.config.enforce_price_sanity and not self._check_price(order, last_prices):
            logger.warning("OrderGuard: price sanity violation for %s", order.symbol)
            return False

        if self.config.enforce_stop_sanity and not self._check_stops(order, last_prices):
            logger.warning("OrderGuard: stop-loss/take-profit sanity violation for %s", order.symbol)
            return False

        return True

    def _check_lot(self, order: OrderRequest) -> bool:
        """
        Enforce min/max lot constraints.
        """
        lot = float(order.lot_size)
        if lot < self.config.min_lot:
            return False
        if lot > self.config.max_lot:
            return False
        return True

    def _check_price(
        self,
        order: OrderRequest,
        last_prices: Dict[str, float],
    ) -> bool:
        """
        Ensure the requested price (if any) is not too far from last known price.
        """
        if order.price is None:
            return True  # market order

        last = float(last_prices.get(order.symbol, 0.0))
        if last <= 0:
            return True  # no reference, allow

        diff = abs(order.price - last) / last
        if diff > self.config.max_slippage_pct:
            return False

        return True

    def _check_stops(
        self,
        order: OrderRequest,
        last_prices: Dict[str, float],
    ) -> bool:
        """
        Ensure SL/TP are within reasonable distances from last price.
        """
        last = float(last_prices.get(order.symbol, 0.0))
        if last <= 0:
            return True  # no reference, allow

        for level in (order.stop_loss, order.take_profit):
            if level is None:
                continue

            dist = abs(level - last) / last
            if dist < self.config.min_stop_distance_pct:
                return False
            if dist > self.config.max_stop_distance_pct:
                return False

        return True
