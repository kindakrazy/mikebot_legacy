# mikebot/api/endpoints/strategy.py

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import Any, Dict

from mikebot.runtime.context import runtime_context

router = APIRouter()


@router.get("/strategy/state")
def get_strategy_state() -> Dict[str, Any]:
    try:
        ctx = runtime_context

        strategy_engine = ctx.get("strategy_engine")
        feature_builder = ctx.get("feature_builder")
        candle_engine = ctx.get("candle_engine")
        minion_registry = ctx.get("minion_registry")
        personality_manager = ctx.get("personality_manager")
        regime_switcher = ctx.get("regime_switcher")
        knowledge_graph = ctx.get("knowledge_graph")

        orch = ctx.get("orchestrator")

        latest_features = getattr(orch, "_latest_features", None)
        last_minion_decisions = getattr(orch, "_last_minion_decisions", None)

        return {
            "timestamp": datetime.utcnow().isoformat(),

            "strategy_engine": {
                "exists": strategy_engine is not None,
                "registry": (
                    strategy_engine.registry.snapshot()
                    if strategy_engine and hasattr(strategy_engine, "registry")
                    else None
                ),
                "toggles": getattr(strategy_engine, "toggles", None),
            },

            "feature_builder": {
                "exists": feature_builder is not None,
                "pipeline": (
                    feature_builder.pipeline_description()
                    if hasattr(feature_builder, "pipeline_description")
                    else None
                ),
            },

            "candle_engine": {
                "exists": candle_engine is not None,
                "latest_snapshot": (
                    candle_engine.get_latest_snapshot()
                    if candle_engine
                    else None
                ),
            },

            "minions": {
                "exists": minion_registry is not None,
                "active_minions": (
                    [m.name for m in minion_registry.iter_active_minions()]
                    if minion_registry
                    else None
                ),
                "last_decisions": last_minion_decisions,
            },

            "personality": {
                "exists": personality_manager is not None,
                "snapshot": (
                    personality_manager.snapshot()
                    if personality_manager and hasattr(personality_manager, "snapshot")
                    else None
                ),
            },

            "regime": {
                "exists": regime_switcher is not None,
                "snapshot": (
                    regime_switcher.snapshot()
                    if regime_switcher and hasattr(regime_switcher, "snapshot")
                    else None
                ),
            },

            "knowledge_graph": {
                "exists": knowledge_graph is not None,
                "summary": (
                    knowledge_graph.summary()
                    if knowledge_graph and hasattr(knowledge_graph, "summary")
                    else None
                ),
            },

            "latest_features": latest_features,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"strategy_state failed: {exc}")