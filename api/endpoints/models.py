# mikebot/api/endpoints/models.py

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import Any, Dict

from mikebot.runtime.context import runtime_context

router = APIRouter()


@router.get("/models/state")
def get_models_state() -> Dict[str, Any]:
    try:
        ctx = runtime_context

        model_registry = ctx.get("model_registry")
        lineage_registry = ctx.get("lineage_registry")
        training_pipeline = ctx.get("training_pipeline")
        training_orchestrator = ctx.get("training_orchestrator")

        # Attempt to extract registry snapshot
        if model_registry is not None:
            try:
                registry_snapshot = model_registry.snapshot()
            except Exception:
                registry_snapshot = None
        else:
            registry_snapshot = None

        # Attempt to extract lineage snapshot
        if lineage_registry is not None:
            try:
                lineage_snapshot = lineage_registry.snapshot()
            except Exception:
                lineage_snapshot = None
        else:
            lineage_snapshot = None

        # Training pipeline metadata
        if training_pipeline is not None:
            try:
                pipeline_info = {
                    "config": getattr(training_pipeline, "config", None),
                    "has_model_factory": hasattr(training_pipeline, "model_factory"),
                }
            except Exception:
                pipeline_info = None
        else:
            pipeline_info = None

        # Training orchestrator metadata
        if training_orchestrator is not None:
            try:
                orchestrator_info = {
                    "active": getattr(training_orchestrator, "active", None),
                    "last_metrics": getattr(training_orchestrator, "last_metrics", None),
                    "last_model_path": getattr(training_orchestrator, "last_model_path", None),
                }
            except Exception:
                orchestrator_info = None
        else:
            orchestrator_info = None

        return {
            "timestamp": datetime.utcnow().isoformat(),

            "model_registry": {
                "exists": model_registry is not None,
                "snapshot": registry_snapshot,
            },

            "lineage_registry": {
                "exists": lineage_registry is not None,
                "snapshot": lineage_snapshot,
            },

            "training_pipeline": {
                "exists": training_pipeline is not None,
                "info": pipeline_info,
            },

            "training_orchestrator": {
                "exists": training_orchestrator is not None,
                "info": orchestrator_info,
            },
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"models_state failed: {exc}")