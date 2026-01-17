from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from .minions_base import OrderRequest
from .survivability import SurvivabilityGuard

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Guardrails configuration
# ---------------------------------------------------------------------------

@dataclass
class GuardrailsConfig:
    """
    Configuration for the guardrails engine.

    Distilled from:
      - guardrails_check
      - guardrail_adaptor
      - guardrails.yml
      - control.json
      - switches.json
    """

    # Global enable/disable
    enabled: bool = True

    # AI master switch (from control.json)
    ai_master: bool = True

    # Hard exposure limits
    max_symbol_exposure_fraction: float = 0.20
    max_total_exposure_fraction: float = 0.50

    # Soft nudging thresholds
    nudge_symbol_exposure_fraction: float = 0.15
    nudge_total_exposure_fraction: float = 0.40

    # Lot size constraints
    min_lot: float = 0.01
    max_lot: float = 50.0

    # Whether to allow soft nudging instead of hard rejection
    enable_nudging: bool = True

    # Volatility-aware nudging
    enable_vol_nudging: bool = True
    vol_nudge_threshold: float = 2.0  # e.g. z-score or normalized vol

    # Session-aware behavior
    cautious_sessions: List[str] = field(default_factory=lambda: ["asia"])
    blocked_sessions: List[str] = field(default_factory=list)

    # SL/TP sanity (relative to last price)
    min_stop_distance_pct: float = 0.002  # 0.2%
    max_stop_distance_pct: float = 0.10   # 10%

    # MT4 bridge gating
    require_bridge_alive: bool = True


# ---------------------------------------------------------------------------
# Guardrails engine
# ---------------------------------------------------------------------------

class Guardrails:
    """
    Guardrails engine for pre-trade enforcement and soft nudging.

    Responsibilities:
      - Enforce hard guardrails (exposure, lot size, AI master)
      - Apply soft nudging (lot size, SL/TP) based on exposure, volatility, session
      - Integrate with survivability guard
      - Optionally gate on MT4 bridge health
      - Provide a clean filter API for order_router
    """

    def __init__(
        self,
        config: GuardrailsConfig,
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
        session_tag: Optional[str] = None,
        bridge_alive: bool = True,
    ) -> List[OrderRequest]:
        """
        Apply guardrails to a list of proposed orders.

        session_tag:
            Optional label like "asia", "london", "ny", etc.
        bridge_alive:
            Optional MT4 bridge health flag.
        """
        if not self.config.enabled:
            return orders

        if not self.config.ai_master:
            logger.warning("Guardrails: AI master disabled → blocking all orders")
            return []

        if self.config.require_bridge_alive and not bridge_alive:
            logger.warning("Guardrails: MT4 bridge not alive → blocking all orders")
            return []

        if session_tag in self.config.blocked_sessions:
            logger.warning("Guardrails: session '%s' blocked → no new orders", session_tag)
            return []

        # Survivability integration
        surv_state = self.survivability.check_survivability(
            account_state=account_state,
            open_positions=open_positions,
            volatility_series=volatility_series,
            loop_iteration=loop_iteration,
        )

        if surv_state.get("safe_mode"):
            logger.warning("Guardrails: SAFE MODE active → blocking all orders")
            return []

        # Simple volatility level (caller can pass z-score series or normalized vol)
        vol_level = None
        if volatility_series is not None and len(volatility_series) > 0:
            vol_level = float(volatility_series[-1])

        filtered: List[OrderRequest] = []
        for o in orders:
            adjusted = self._apply_guardrails(
                order=o,
                account_state=account_state,
                open_positions=open_positions,
                last_prices=last_prices,
                vol_level=vol_level,
                session_tag=session_tag,
            )
            if adjusted is not None:
                filtered.append(adjusted)

        return filtered

    # ----------------------------------------------------------------------
    # Core guardrail logic
    # ----------------------------------------------------------------------

    def _apply_guardrails(
        self,
        order: OrderRequest,
        account_state: Dict[str, Any],
        open_positions: List[Dict[str, Any]],
        last_prices: Dict[str, float],
        vol_level: Optional[float],
        session_tag: Optional[str],
    ) -> Optional[OrderRequest]:
        """
        Apply hard and soft guardrails to a single order.
        """
        # Hard lot limits
        if not self._check_lot(order):
            logger.warning("Guardrails: lot size violation for %s", order.symbol)
            return None

        # Exposure limits
        exposure_status = self._check_exposure(order, account_state, open_positions, last_prices)

        if exposure_status == "reject":
            logger.warning("Guardrails: exposure violation for %s", order.symbol)
            return None

        adjusted = order

        # Soft nudging based on exposure
        if exposure_status == "nudge" and self.config.enable_nudging:
            adjusted = self._nudge_order(
                adjusted,
                last_prices=last_prices,
                vol_level=vol_level,
                session_tag=session_tag,
            )

        # SL/TP sanity nudging
        adjusted = self._nudge_stops(adjusted, last_prices)

        return adjusted

    # ----------------------------------------------------------------------
    # Lot size guardrails
    # ----------------------------------------------------------------------

    def _check_lot(self, order: OrderRequest) -> bool:
        lot = float(order.lot_size)
        if lot < self.config.min_lot:
            return False
        if lot > self.config.max_lot:
            return False
        return True

    # ----------------------------------------------------------------------
    # Exposure guardrails
    # ----------------------------------------------------------------------

    def _check_exposure(
        self,
        order: OrderRequest,
        account_state: Dict[str, Any],
        open_positions: List[Dict[str, Any]],
        last_prices: Dict[str, float],
    ) -> str:
        """
        Returns:
            "ok"     → allow
            "nudge"  → allow but reduce lot
            "reject" → block order
        """
        equity = float(account_state.get("equity", 0.0))
        if equity <= 0:
            return "reject"

        symbol = order.symbol
        price = float(last_prices.get(symbol, 0.0))
        if price <= 0:
            return "ok"

        new_notional = order.lot_size * price

        # Compute existing exposure
        symbol_exp = 0.0
        total_exp = 0.0

        for pos in open_positions:
            sym = pos.get("symbol")
            lots = float(pos.get("lots", 0.0))
            p = float(pos.get("price", 0.0))
            if p <= 0:
                continue

            notional = abs(lots * p)
            total_exp += notional

            if sym == symbol:
                symbol_exp += notional

        # Add new exposure
        symbol_exp_new = symbol_exp + abs(new_notional)
        total_exp_new = total_exp + abs(new_notional)

        # Hard limits
        if symbol_exp_new / equity > self.config.max_symbol_exposure_fraction:
            return "reject"
        if total_exp_new / equity > self.config.max_total_exposure_fraction:
            return "reject"

        # Soft nudging thresholds
        if (
            symbol_exp_new / equity > self.config.nudge_symbol_exposure_fraction
            or total_exp_new / equity > self.config.nudge_total_exposure_fraction
        ):
            return "nudge"

        return "ok"

    # ----------------------------------------------------------------------
    # Soft nudging (lot size + volatility/session aware)
    # ----------------------------------------------------------------------

    def _nudge_order(
        self,
        order: OrderRequest,
        last_prices: Dict[str, float],
        vol_level: Optional[float],
        session_tag: Optional[str],
    ) -> OrderRequest:
        """
        Reduce lot size (and optionally more) to stay within soft guardrail thresholds.
        """
        factor = 0.5

        # Volatility-aware extra caution
        if self.config.enable_vol_nudging and vol_level is not None:
            if abs(vol_level) >= self.config.vol_nudge_threshold:
                factor *= 0.5  # cut again in high vol

        # Session-aware extra caution
        if session_tag and session_tag in self.config.cautious_sessions:
            factor *= 0.7

        nudged_lot = float(order.lot_size) * factor
        nudged_lot = max(nudged_lot, self.config.min_lot)

        logger.info(
            "Guardrails: nudging order %s from %.4f → %.4f (vol=%.3f, session=%s)",
            order.symbol,
            order.lot_size,
            nudged_lot,
            vol_level if vol_level is not None else float("nan"),
            session_tag,
        )

        return order.with_lot_size(nudged_lot)

    # ----------------------------------------------------------------------
    # SL/TP nudging
    # ----------------------------------------------------------------------

    def _nudge_stops(self, order: OrderRequest, last_prices: Dict[str, float]) -> OrderRequest:
        """
        Ensure SL/TP are within reasonable distances from last price.
        If too tight → push further away.
        If too far → pull closer but still within max distance.
        """
        last = float(last_prices.get(order.symbol, 0.0))
        if last <= 0:
            return order

        sl = order.stop_loss
        tp = order.take_profit

        def _adjust(level: Optional[float]) -> Optional[float]:
            if level is None:
                return None
            dist = abs(level - last) / last
            if dist < self.config.min_stop_distance_pct:
                # push further away
                sign = 1.0 if level > last else -1.0
                return last * (1.0 + sign * self.config.min_stop_distance_pct)
            if dist > self.config.max_stop_distance_pct:
                # pull closer
                sign = 1.0 if level > last else -1.0
                return last * (1.0 + sign * self.config.max_stop_distance_pct)
            return level

        new_sl = _adjust(sl)
        new_tp = _adjust(tp)

        if new_sl != sl or new_tp != tp:
            logger.info(
                "Guardrails: nudging SL/TP for %s (SL: %s→%s, TP: %s→%s)",
                order.symbol,
                sl,
                new_sl,
                tp,
                new_tp,
            )

        return OrderRequest(
            symbol=order.symbol,
            side=order.side,
            lot_size=order.lot_size,
            price=order.price,
            stop_loss=new_sl,
            take_profit=new_tp,
            comment=order.comment,
        )