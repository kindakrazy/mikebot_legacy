# mikebot/api/endpoints/system_state.py

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import Any, Dict

from mikebot.runtime.context import runtime_context

router = APIRouter()


@router.get("/system/state")
def get_system_state() -> Dict[str, Any]:
    try:
        ctx = runtime_context

        orch = ctx.get("orchestrator")
        candle_engine = ctx.get("candle_engine")
        feature_builder = ctx.get("feature_builder")
        strategy_engine = ctx.get("strategy_engine")
        minion_registry = ctx.get("minion_registry")
        order_router = ctx.get("order_router")
        telemetry = ctx.get("telemetry")
        personality_manager = ctx.get("personality_manager")
        survivability_guard = ctx.get("survivability_guard")
        portfolio_optimizer = ctx.get("portfolio_optimizer")
        guardrails = ctx.get("guardrails")
        health_monitor = ctx.get("health_monitor")
        regime_detector = ctx.get("regime_detector")
        learner = ctx.get("learner")
        neural_layer = ctx.get("neural_layer")
        regime_switcher = ctx.get("regime_switcher")
        knowledge_graph = ctx.get("knowledge_graph")
        market_server = ctx.get("market_server")

        training_orchestrator = ctx.get("training_orchestrator")
        training_pipeline = ctx.get("training_pipeline")
        training_state = ctx.get("training_state")

        return {
            "timestamp": datetime.utcnow().isoformat(),

            "orchestrator": {
                "exists": orch is not None,
                "state": orch.state.__dict__ if hasattr(orch, "state") else None,
                "current_regime": getattr(orch, "current_regime", None),
                "last_minion_decisions": getattr(orch, "_last_minion_decisions", None),
                "last_orders": getattr(orch, "_last_orders", None),
                "last_survivability_state": getattr(orch, "_last_survivability_state", None),
                "latest_features": getattr(orch, "_latest_features", None),
            },

            "candle_engine": {
                "exists": candle_engine is not None,
                "snapshot": candle_engine.get_latest_snapshot() if candle_engine else None,
            },

            "feature_builder": {
                "exists": feature_builder is not None,
            },

            "strategy_engine": {
                "exists": strategy_engine is not None,
                "registry": (
                    strategy_engine.registry.snapshot()
                    if hasattr(strategy_engine, "registry")
                    else None
                ),
            },

            "minion_registry": {
                "exists": minion_registry is not None,
                "active_minions": (
                    [m.name for m in minion_registry.iter_active_minions()]
                    if minion_registry
                    else None
                ),
            },

            "order_router": {
                "exists": order_router is not None,
                "account_state": (
                    order_router.get_account_state() if order_router else None
                ),
                "open_positions": (
                    order_router.get_open_positions() if order_router else None
                ),
                "snapshot": (
                    order_router.snapshot() if hasattr(order_router, "snapshot") else None
                ),
            },

            "telemetry": {
                "exists": telemetry is not None,
            },

            "personality_manager": {
                "exists": personality_manager is not None,
                "snapshot": (
                    personality_manager.snapshot()
                    if hasattr(personality_manager, "snapshot")
                    else None
                ),
            },

            "survivability_guard": {
                "exists": survivability_guard is not None,
            },

            "portfolio_optimizer": {
                "exists": portfolio_optimizer is not None,
            },

            "guardrails": {
                "exists": guardrails is not None,
            },

            "health_monitor": {
                "exists": health_monitor is not None,
                "snapshot": (
                    health_monitor.snapshot()
                    if hasattr(health_monitor, "snapshot")
                    else None
                ),
            },

            "regime_detector": {
                "exists": regime_detector is not None,
            },

            "learner": {
                "exists": learner is not None,
            },

            "neural_layer": {
                "exists": neural_layer is not None,
            },

            "regime_switcher": {
                "exists": regime_switcher is not None,
                "snapshot": (
                    regime_switcher.snapshot()
                    if hasattr(regime_switcher, "snapshot")
                    else None
                ),
            },

            "knowledge_graph": {
                "exists": knowledge_graph is not None,
            },

            "market_server": {
                "exists": market_server is not None,
            },

            "training": {
                "training_orchestrator_exists": training_orchestrator is not None,
                "training_pipeline_exists": training_pipeline is not None,
                "training_state": training_state,
            },
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"system_state failed: {exc}")