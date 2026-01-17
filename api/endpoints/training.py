# mikebot/api/endpoints/training.py

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import Any, Dict

from mikebot.runtime.context import runtime_context

router = APIRouter()


@router.get("/training/state")
def get_training_state() -> Dict[str, Any]:
    try:
        ctx = runtime_context

        training_orchestrator = ctx.get("training_orchestrator")
        training_pipeline = ctx.get("training_pipeline")
        training_state = ctx.get("training_state")

        model_registry = ctx.get("model_registry")
        lineage_registry = ctx.get("lineage_registry")

        # Training orchestrator snapshot
        if training_orchestrator is not None:
            try:
                orchestrator_info = {
                    "active": getattr(training_orchestrator, "active", None),
                    "current_epoch": getattr(training_orchestrator, "current_epoch", None),
                    "current_batch": getattr(training_orchestrator, "current_batch", None),
                    "last_metrics": getattr(training_orchestrator, "last_metrics", None),
                    "last_model_path": getattr(training_orchestrator, "last_model_path", None),
                    "last_error": getattr(training_orchestrator, "last_error", None),
                }
            except Exception:
                orchestrator_info = None
        else:
            orchestrator_info = None

        # Training pipeline snapshot
        if training_pipeline is not None:
            try:
                pipeline_info = {
                    "config": getattr(training_pipeline, "config", None),
                    "has_model_factory": hasattr(training_pipeline, "model_factory"),
                    "has_dataloader": hasattr(training_pipeline, "dataloader"),
                    "has_trainer": hasattr(training_pipeline, "trainer"),
                }
            except Exception:
                pipeline_info = None
        else:
            pipeline_info = None

        # Model registry snapshot
        if model_registry is not None:
            try:
                registry_snapshot = model_registry.snapshot()
            except Exception:
                registry_snapshot = None
        else:
            registry_snapshot = None

        # Lineage registry snapshot
        if lineage_registry is not None:
            try:
                lineage_snapshot = lineage_registry.snapshot()
            except Exception:
                lineage_snapshot = None
        else:
            lineage_snapshot = None

        return {
            "timestamp": datetime.utcnow().isoformat(),

            "training_orchestrator": {
                "exists": training_orchestrator is not None,
                "info": orchestrator_info,
            },

            "training_pipeline": {
                "exists": training_pipeline is not None,
                "info": pipeline_info,
            },

            "training_state": training_state,

            "model_registry": {
                "exists": model_registry is not None,
                "snapshot": registry_snapshot,
            },

            "lineage_registry": {
                "exists": lineage_registry is not None,
                "snapshot": lineage_snapshot,
            },
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"training_state failed: {exc}")