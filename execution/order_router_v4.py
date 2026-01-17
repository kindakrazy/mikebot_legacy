# mikebot/execution/order_router_v4.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List
from pathlib import Path

from mikebot.execution.execution_context_v4 import (
    BrokerOrderV4,
    BrokerPositionV4,
    BrokerAccountStateV4,
    MarketTickV4,
)


@dataclass
class OrderResultV4:
    """
    Result of an order routing operation.
    """
    success: bool
    order_id: Optional[str] = None
    message: Optional[str] = None
    raw: Dict[str, Any] = None


class OrderRouterV4:
    """
    Abstract, broker-agnostic order routing layer.

    This class defines the canonical interface for:
    - market orders
    - limit orders
    - stop orders
    - order cancellation
    - position closure

    Concrete broker adapters should subclass this and implement the
    protected _send_* methods.

    The execution engine interacts ONLY with this interface.
    """

    def __init__(self, *, dry_run: bool = False) -> None:
        self.dry_run = dry_run

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_market_order(
        self,
        symbol: str,
        side: str,
        size: float,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OrderResultV4:
        """
        Send a market order.
        """
        if self.dry_run:
            return OrderResultV4(
                success=True,
                order_id="dryrun-market",
                message="Dry-run market order accepted",
                raw={"symbol": symbol, "side": side, "size": size},
            )

        return self._send_market_order(symbol, side, size, metadata or {})

    def send_limit_order(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OrderResultV4:
        """
        Send a limit order.
        """
        if self.dry_run:
            return OrderResultV4(
                success=True,
                order_id="dryrun-limit",
                message="Dry-run limit order accepted",
                raw={"symbol": symbol, "side": side, "size": size, "price": price},
            )

        return self._send_limit_order(symbol, side, size, price, metadata or {})

    def send_stop_order(
        self,
        symbol: str,
        side: str,
        size: float,
        stop_price: float,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OrderResultV4:
        """
        Send a stop order.
        """
        if self.dry_run:
            return OrderResultV4(
                success=True,
                order_id="dryrun-stop",
                message="Dry-run stop order accepted",
                raw={"symbol": symbol, "side": side, "size": size, "stop_price": stop_price},
            )

        return self._send_stop_order(symbol, side, size, stop_price, metadata or {})

    def cancel_order(self, order_id: str) -> OrderResultV4:
        """
        Cancel a pending order.
        """
        if self.dry_run:
            return OrderResultV4(
                success=True,
                order_id=order_id,
                message="Dry-run cancellation accepted",
            )

        return self._cancel_order(order_id)

    def close_position(self, position_id: str) -> OrderResultV4:
        """
        Close an open position.
        """
        if self.dry_run:
            return OrderResultV4(
                success=True,
                order_id=position_id,
                message="Dry-run position close accepted",
            )

        return self._close_position(position_id)

    # ------------------------------------------------------------------
    # Protected broker-specific methods
    # ------------------------------------------------------------------

    def _send_market_order(
        self,
        symbol: str,
        side: str,
        size: float,
        metadata: Dict[str, Any],
    ) -> OrderResultV4:
        raise NotImplementedError("Broker adapter must implement _send_market_order")

    def _send_limit_order(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        metadata: Dict[str, Any],
    ) -> OrderResultV4:
        raise NotImplementedError("Broker adapter must implement _send_limit_order")

    def _send_stop_order(
        self,
        symbol: str,
        side: str,
        size: float,
        stop_price: float,
        metadata: Dict[str, Any],
    ) -> OrderResultV4:
        raise NotImplementedError("Broker adapter must implement _send_stop_order")

    def _cancel_order(self, order_id: str) -> OrderResultV4:
        raise NotImplementedError("Broker adapter must implement _cancel_order")

    def _close_position(self, position_id: str) -> OrderResultV4:
        raise NotImplementedError("Broker adapter must implement _close_position")