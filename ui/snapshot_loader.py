# mikebot/ui/snapshot_loader.py
"""
Snapshot loader for Mikebot Studio.

Responsible for:
    - Validating snapshot structure
    - Populating mikebot.ui.global_state
    - Setting mode to "snapshot"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from mikebot.ui import global_state


def load_snapshot_file(path: str | Path) -> Dict[str, Any]:
    """
    Load a snapshot from a JSON file and apply it to global_state.
    Returns a dict:
        - on success: { "ok": True }
        - on failure: { "error": "..." }
    """
    p = Path(path)
    if not p.exists():
        return {"error": f"Snapshot file not found: {p}"}

    try:
        raw = p.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception as exc:
        return {"error": f"Failed to read/parse snapshot: {exc}"}

    return apply_snapshot_dict(data)


def apply_snapshot_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply a snapshot dict to global_state.
    Expected structure:
        {
          "system": {...},
          "strategy": {...},
          "models": {...},
          "experience": {...},
          "training": {...}
        }
    """
    if not isinstance(data, dict):
        return {"error": "Snapshot root must be a dict"}

    # Extract sections (missing sections are allowed, they become None)
    system = data.get("system")
    strategy = data.get("strategy")
    models = data.get("models")
    experience = data.get("experience")
    training = data.get("training")

    # Basic shape validation (soft)
    # We don't enforce schemas here, just type sanity.
    for key, section in [
        ("system", system),
        ("strategy", strategy),
        ("models", models),
        ("experience", experience),
        ("training", training),
    ]:
        if section is not None and not isinstance(section, dict):
            return {"error": f"Snapshot section '{key}' must be a dict or null"}

    # Apply to global_state
    global_state.set_system_state(system)
    global_state.set_strategy_state(strategy)
    global_state.set_models_state(models)
    global_state.set_experience_state(experience)
    global_state.set_training_state(training)
    global_state.set_mode("snapshot")

    return {"ok": True}