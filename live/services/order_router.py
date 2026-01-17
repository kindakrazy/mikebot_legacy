from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

import asyncio
import pandas as pd

from mikebot.live.orchestrator.config import LiveConfig
from mikebot.minions.minions_base import OrderRequest, OrderSide

from mikebot.minions.guardrails import Guardrails, GuardrailsConfig
from mikebot.minions.survivability import SurvivabilityGuard, SurvivabilityConfig
from mikebot.minions.max_lot_calc import MaxLotCalculator
from mikebot.minions.order_queue import OrderQueue
from mikebot.live.services.telemetry import TelemetryService

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OrderRouter configuration
# ---------------------------------------------------------------------------

@dataclass
class OrderRouterConfig:
    mt4_orders_csv: Path
    mt4_status_csv: Path
    jsonl_export_path: Path
    max_lot_calc: Dict[str, Any]
    guardrails: Dict[str, Any]
    queue_size: int = 1000


# ---------------------------------------------------------------------------
# OrderRouter
# ---------------------------------------------------------------------------

class OrderRouter:
    """
    Canonical mikebot v3 order routing engine.

    Responsibilities:
      - Apply guardrails (risk, exposure, volatility, session)
      - Enforce max-lot sizing
      - Queue validated orders
      - Export to MT4 (CSV + JSONL)
      - Emit telemetry
    """

    def __init__(
        self,
        config: OrderRouterConfig,
        telemetry: TelemetryService,
    ) -> None:
        self.config = config
        self.telemetry = telemetry
        self.orders: Dict[str, OrderRequest] = {}
        self.queue = OrderQueue(max_size=config.queue_size)

        survivability = SurvivabilityGuard(
            SurvivabilityConfig()  # can be exposed later if needed
        )

        self.guardrails = Guardrails(
            GuardrailsConfig(**config.guardrails),
            survivability,
        )

        self.max_lot_calc = MaxLotCalculator(config.max_lot_calc)
        self.execution_bridge = None

        # Live snapshots from ExecutionBridgeEA
        self._account_state: Dict[str, Any] = {}
        self._open_positions: List[Dict[str, Any]] = []

        self._ensure_paths()

    # ----------------------------------------------------------------------
    # Construction helpers
    # ----------------------------------------------------------------------

    @classmethod
    def from_config(cls, live_cfg: LiveConfig, telemetry: TelemetryService) -> OrderRouter:
        root = live_cfg.root

        return cls(
            config=OrderRouterConfig(
                mt4_orders_csv=root / "MQL4" / "Files" / "hs_orders.csv",
                mt4_status_csv=root / "MQL4" / "Files" / "hs_status.csv",
                jsonl_export_path=root / "export" / "mt4" / "highstrike_orders.jsonl",
                max_lot_calc=live_cfg.risk_caps,
                guardrails=live_cfg.guardrails,
                queue_size=live_cfg.control.get("order_queue_size", 1000),
            ),
            telemetry=telemetry,
        )

    def _ensure_paths(self) -> None:
        self.config.jsonl_export_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.mt4_orders_csv.parent.mkdir(parents=True, exist_ok=True)

    def attach_execution_bridge(self, bridge):
        """
        Attach a MikebotBridgeServer instance so this router can send
        live execution commands to MT4.
        """
        self.execution_bridge = bridge
        print("[OrderRouter] Execution bridge attached")

    def handle_execution_event(self, event: dict):
        """
        Process messages coming from the ExecutionBridgeEA.

        EA sends:
          - type: "event"     (order_opened / order_closed / order_modified / trailing_started)
          - type: "account"   (account snapshot)
          - type: "positions" (open positions snapshot)
          - type: "error"     (error messages)

        This method:
          - Updates internal order state
          - Tracks fill price and fill time
          - Maintains account + positions snapshots
          - Emits telemetry
        """
        try:
            print("[OrderRouter] Execution event:", event)

            msg_type = event.get("type")

            # --------------------------------------------------------------
            # 0. Account snapshot
            # --------------------------------------------------------------
            if msg_type == "account":
                bal = float(event.get("balance", 0.0))
                eq = float(event.get("equity", 0.0))
                mar = float(event.get("margin", 0.0))
                free = float(event.get("free_margin", 0.0))

                self._account_state = {
                    "balance": bal,
                    "equity": eq,
                    "margin": mar,
                    "free_margin": free,
                }

                # Optional: emit telemetry
                self.telemetry.emit_account_snapshot(self._account_state)
                return

            # --------------------------------------------------------------
            # 1. Positions snapshot
            # --------------------------------------------------------------
            if msg_type == "positions":
                positions = event.get("positions") or []
                if not isinstance(positions, list):
                    positions = []

                # Normalize positions into a cockpit-friendly schema
                norm_positions: List[Dict[str, Any]] = []
                symbol = event.get("symbol")

                for p in positions:
                    try:
                        norm_positions.append(
                            {
                                "ticket": p.get("ticket"),
                                "symbol": symbol,
                                "side": p.get("type"),
                                "lots": float(p.get("lots", 0.0)),
                                "open_price": float(p.get("open_price", 0.0)),
                                "sl": float(p.get("sl", 0.0)),
                                "tp": float(p.get("tp", 0.0)),
                                "profit": float(p.get("profit", 0.0)),
                            }
                        )
                    except Exception:
                        continue

                self._open_positions = norm_positions
                self.telemetry.emit_positions_snapshot(self._open_positions)
                return

            # --------------------------------------------------------------
            # 2. Error messages
            # --------------------------------------------------------------
            if msg_type == "error":
                where = event.get("where")
                code = event.get("code")
                message = event.get("message")
                cid = event.get("correlation_id")
                log.warning(
                    "ExecutionBridgeEA error in %s (code=%s, cid=%s): %s",
                    where,
                    code,
                    cid,
                    message,
                )
                self.telemetry.emit_error("execution_bridge_error", message or str(event))
                return

            # --------------------------------------------------------------
            # 3. Order events
            # --------------------------------------------------------------
            if msg_type not in (None, "event", "execution"):
                # Unknown type, ignore
                log.debug("Ignoring execution message with type=%s → %s", msg_type, event)
                return

            order_id = event.get("order_id") or event.get("ticket")
            if not order_id:
                log.warning("Execution event missing order_id/ticket → %s", event)
                return

            # Find the order if we have it
            order = self.orders.get(str(order_id))

            evt_type = event.get("event")
            fill_price = event.get("price")
            fill_time = event.get("timestamp")
            reason = event.get("reason")

            # If we don't know this order yet, we still emit telemetry but
            # we don't try to mutate a non-existent OrderRequest.
            if order is None:
                self.telemetry.emit_execution_event(
                    {
                        "order_id": order_id,
                        "event": evt_type,
                        "price": fill_price,
                        "timestamp": fill_time,
                        "reason": reason,
                    }
                )
                return

            updated = order

            # Fill price
            if fill_price is not None:
                try:
                    updated = updated.with_fill_price(float(fill_price))
                except Exception:
                    pass

            # Fill timestamp
            if fill_time:
                try:
                    ts = datetime.fromisoformat(str(fill_time).replace("Z", "+00:00"))
                    updated = updated.with_fill_timestamp(ts)
                except Exception:
                    log.warning("Invalid timestamp in execution event → %s", fill_time)

            # Map EA event names to internal status
            status = None
            if evt_type in ("opened", "order_opened"):
                status = "open"
            elif evt_type in ("filled",):
                status = "filled"
            elif evt_type in ("closed", "order_closed"):
                status = "closed"
            elif evt_type in ("rejected",):
                status = "rejected"
            elif evt_type in ("order_modified", "trailing_started"):
                # no status change, but we still keep updated fill/sl/tp if any
                status = None

            if status is not None:
                try:
                    updated = updated.with_status(status)
                except Exception:
                    pass

            self.orders[str(order_id)] = updated

            self.telemetry.emit_execution_event(
                {
                    "order_id": order_id,
                    "event": evt_type,
                    "price": fill_price,
                    "timestamp": fill_time,
                    "reason": reason,
                }
            )

            # Optional: audit log
            self._export_execution_jsonl(event)

        except Exception as exc:
            log.exception("Failed to process execution event: %s", exc)
            self.telemetry.emit_error("execution_event_error", str(exc))

    def _export_execution_jsonl(self, event: dict) -> None:
        """
        Append execution events to a JSONL audit log.
        """
        path = self.config.jsonl_export_path.parent / "execution_events.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def route_order(self, order: OrderRequest, ctx) -> None:
        """
        Main entry point for routing a single order.

        Steps:
          1. Guardrails (risk, exposure, volatility, session)
          2. Max lot sizing
          3. Queue insertion
          4. MT4 export (CSV + JSONL)
          5. Telemetry
          6. Live execution (if bridge attached)
        """
        try:
            # 1. Guardrails (list-based API)
            guarded_list = self.guardrails.filter_orders(
                orders=[order],
                account_state=ctx.account_state,
                open_positions=ctx.open_positions,
                last_prices=ctx.last_prices,
                loop_iteration=ctx.loop_iteration,
                volatility_series=ctx.volatility_series,
                session_tag=ctx.session_tag,
                bridge_alive=True,
            )

            if not guarded_list:
                log.warning("OrderRouter: order blocked by guardrails → %s", order)
                return

            guarded = guarded_list[0]

            # 2. Max lot sizing
            sized = self._apply_max_lot(guarded, ctx)
            self.orders[str(sized.id)] = sized

            # 3. Queue insertion
            self.queue.push(sized)

            # 4. MT4 export
            self._export_to_mt4(sized)
            self._export_jsonl(sized)

            # 5. Telemetry
            self.telemetry.emit_order_routed(sized)

            # 6. Live execution (fire-and-forget)
            self._send_to_execution_bridge(sized)

        except Exception as exc:
            msg = f"Order routing failed for {order}: {exc!r}"
            log.exception(msg)
            self.telemetry.emit_error("order_routing_error", msg)
            raise

    def snapshot(self):
        """
        Return a cockpit‑compatible list of order dictionaries.
        """
        out = []

        for order in self.orders.values():
            out.append(
                {
                    "id": order.id,
                    "symbol": order.symbol,
                    "side": order.side,
                    "size": order.size,
                    "price": order.price,
                    "status": order.status,
                    "timestamp": (
                        order.timestamp.isoformat()
                        if hasattr(order, "timestamp") and order.timestamp
                        else None
                    ),
                }
            )

        return out

    # ----------------------------------------------------------------------
    # Live execution bridge
    # ----------------------------------------------------------------------

    def _send_to_execution_bridge(self, order: OrderRequest) -> None:
        """
        Fire-and-forget send of a live order command to the ExecutionBridgeEA,
        if an execution bridge is attached.
        """
        if self.execution_bridge is None:
            return

        try:
            cmd = {
                "type": "cmd",
                "cmd": "order",
                "symbol": order.symbol,
                "side": order.side.value if isinstance(order.side, OrderSide) else str(order.side),
                "lot": order.lot_size,
                "price": order.price or 0.0,
                "sl": order.stop_loss or 0.0,
                "tp": order.take_profit or 0.0,
                "comment": order.comment or "",
            }

            asyncio.run_coroutine_threadsafe(
                self.execution_bridge.send_command(order.symbol, cmd),
                self.execution_bridge.loop,
            )

        except Exception as exc:
            log.exception("Failed to send order to execution bridge: %s", exc)
            self.telemetry.emit_error("execution_bridge_error", str(exc))

    # ----------------------------------------------------------------------
    # Max lot sizing
    # ----------------------------------------------------------------------

    def _apply_max_lot(self, order: OrderRequest, ctx) -> OrderRequest:
        """
        Apply max lot sizing rules.
        """
        max_lot = self.max_lot_calc.compute(
            symbol=order.symbol,
            account_state=ctx.account_state,
            open_positions=ctx.open_positions,
        )

        if abs(order.lot_size) > max_lot:
            order = order.with_lot_size(max_lot * (1 if order.lot_size > 0 else -1))

        return order

    # ----------------------------------------------------------------------
    # MT4 export (CSV)
    # ----------------------------------------------------------------------

    def _export_to_mt4(self, order: OrderRequest) -> None:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": order.symbol,
            "side": order.side.value if isinstance(order.side, OrderSide) else str(order.side),
            "lot": order.lot_size,
            "price": order.price or 0.0,
            "sl": order.stop_loss or 0.0,
            "tp": order.take_profit or 0.0,
            "comment": order.comment or "",
        }

        file_exists = self.config.mt4_orders_csv.exists()
        with self.config.mt4_orders_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "symbol",
                    "side",
                    "lot",
                    "price",
                    "sl",
                    "tp",
                    "comment",
                ],
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    # ----------------------------------------------------------------------
    # JSONL export
    # ----------------------------------------------------------------------

    def _export_jsonl(self, order: OrderRequest) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": order.symbol,
            "side": order.side.value if isinstance(order.side, OrderSide) else str(order.side),
            "lot": order.lot_size,
            "price": order.price,
            "stop_loss": order.stop_loss,
            "take_profit": order.take_profit,
            "comment": order.comment,
        }

        with self.config.jsonl_export_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    # ----------------------------------------------------------------------
    # Account state + positions
    # ----------------------------------------------------------------------

    def get_account_state(self) -> Dict[str, Any]:
        """
        Prefer live snapshots from ExecutionBridgeEA; fall back to CSV if needed.
        """
        if self._account_state:
            return dict(self._account_state)

        if not self.config.mt4_status_csv.exists():
            return {"balance": 0.0, "equity": 0.0, "margin": 0.0}

        df = pd.read_csv(self.config.mt4_status_csv)
        if df.empty:
            return {"balance": 0.0, "equity": 0.0, "margin": 0.0}

        row = df.iloc[-1]
        return {
            "balance": float(row.get("balance", 0.0)),
            "equity": float(row.get("equity", 0.0)),
            "margin": float(row.get("margin", 0.0)),
        }

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Return the latest open positions snapshot from ExecutionBridgeEA.
        """
        return list(self._open_positions)