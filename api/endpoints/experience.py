# mikebot/api/endpoints/experience.py

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import Any, Dict

from mikebot.runtime.context import runtime_context

router = APIRouter()


@router.get("/experience/state")
def get_experience_state() -> Dict[str, Any]:
    try:
        ctx = runtime_context

        experience_store = ctx.get("experience_store")
        feature_aggregator = ctx.get("feature_aggregator")
        orchestrator = ctx.get("orchestrator")

        # Multi‑TF data (if orchestrator populated it)
        latest_features = getattr(orchestrator, "_latest_features", None)

        # Experience store summary
        if experience_store is not None:
            try:
                store_summary = experience_store.summary()
            except Exception:
                store_summary = None
        else:
            store_summary = None

        # Feature aggregator summary
        if feature_aggregator is not None:
            try:
                aggregator_summary = feature_aggregator.summary()
            except Exception:
                aggregator_summary = None
        else:
            aggregator_summary = None

        return {
            "timestamp": datetime.utcnow().isoformat(),

            "experience_store": {
                "exists": experience_store is not None,
                "summary": store_summary,
                "available_symbols": (
                    experience_store.available_symbols()
                    if experience_store and hasattr(experience_store, "available_symbols")
                    else None
                ),
                "available_timeframes": (
                    experience_store.available_timeframes()
                    if experience_store and hasattr(experience_store, "available_timeframes")
                    else None
                ),
            },

            "feature_aggregator": {
                "exists": feature_aggregator is not None,
                "summary": aggregator_summary,
            },

            "latest_features": latest_features,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"experience_state failed: {exc}")