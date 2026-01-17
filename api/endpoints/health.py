# mikebot/api/endpoints/health.py

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import Any, Dict

from mikebot.runtime.context import runtime_context

router = APIRouter()


@router.get("/health")
def get_health_state() -> Dict[str, Any]:
    try:
        ctx = runtime_context

        health_monitor = ctx.get("health_monitor")
        minion_registry = ctx.get("minion_registry")
        order_router = ctx.get("order_router")
        orchestrator = ctx.get("orchestrator")

        # Health monitor snapshot
        if health_monitor is not None:
            try:
                health_snapshot = health_monitor.snapshot()
            except Exception:
                health_snapshot = None
        else:
            health_snapshot = None

        # Minion health
        if minion_registry is not None:
            try:
                minion_health = {
                    m.name: getattr(m, "health", None)
                    for m in minion_registry.iter_active_minions()
                }
            except Exception:
                minion_health = None
        else:
            minion_health = None

        # Order router health
        if order_router is not None:
            try:
                router_health = {
                    "account_state": order_router.get_account_state(),
                    "open_positions": order_router.get_open_positions(),
                    "snapshot": (
                        order_router.snapshot()
                        if hasattr(order_router, "snapshot")
                        else None
                    ),
                }
            except Exception:
                router_health = None
        else:
            router_health = None

        # Orchestrator heartbeat
        orchestrator_state = None
        if orchestrator is not None:
            try:
                orchestrator_state = {
                    "last_tick": getattr(orchestrator, "last_tick_time", None),
                    "last_loop": getattr(orchestrator, "last_loop_time", None),
                    "current_regime": getattr(orchestrator, "current_regime", None),
                }
            except Exception:
                orchestrator_state = None

        return {
            "timestamp": datetime.utcnow().isoformat(),

            "health_monitor": {
                "exists": health_monitor is not None,
                "snapshot": health_snapshot,
            },

            "minions": {
                "exists": minion_registry is not None,
                "health": minion_health,
            },

            "order_router": {
                "exists": order_router is not None,
                "health": router_health,
            },

            "orchestrator": {
                "exists": orchestrator is not None,
                "state": orchestrator_state,
            },
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"health_state failed: {exc}")