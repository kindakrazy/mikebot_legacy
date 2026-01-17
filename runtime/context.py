# mikebot/runtime/context.py

"""
Global runtime context for the Mikebot system.

This module provides a single shared dictionary, `runtime_context`,
which is populated by the orchestrator process and read by the API
and UI layers.

All assignments are direct. No helpers. No classes. No accessors.
No initialization beyond an empty dict. Keys are added as the
orchestrator builds subsystems.

Training state uses the nested structure requested.
"""

runtime_context = {}

# Predefine the nested training_state structure so it always exists.
runtime_context["training_state"] = {
    "status": {
        "active": False,
        "started_at": None,
        "updated_at": None,
    },
    "progress": {
        "epoch": 0,
        "batch": 0,
        "percent": 0.0,
    },
    "metrics": {},
    "errors": {
        "last_error": None,
        "history": [],
    },
}