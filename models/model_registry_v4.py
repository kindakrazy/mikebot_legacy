# mikebot/models/model_registry_v4.py
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class ModelRecordV4:
    """
    Canonical metadata record for a v4 model.
    """
    model_id: str
    experiment_name: str
    model_type: str
    created_at: str
    metrics: Dict[str, float]
    config: Dict[str, Any]
    artifact_path: str
    lineage: Dict[str, Any]


class ModelRegistryV4:
    """
    Lightweight, file‑based model registry for v4.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.registry_dir = self.root / "registry"
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------

    def create_record(
        self,
        *,
        experiment_name: str,
        model_type: str,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        artifact_path: str,
        lineage: Optional[Dict[str, Any]] = None,
    ) -> ModelRecordV4:
        model_id = str(uuid.uuid4())
        record = ModelRecordV4(
            model_id=model_id,
            experiment_name=experiment_name,
            model_type=model_type,
            created_at=pd.Timestamp.utcnow().isoformat(),
            metrics=dict(metrics),
            config=dict(config),
            artifact_path=str(artifact_path),
            lineage=dict(lineage) if lineage else {},
        )

        self._write_record(record)
        return record

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def load_record(self, model_id: str) -> ModelRecordV4:
        path = self.registry_dir / f"{model_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Model record not found: {model_id}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return ModelRecordV4(**data)

    def list_records(self) -> Dict[str, ModelRecordV4]:
        out: Dict[str, ModelRecordV4] = {}
        for file in self.registry_dir.glob("*.json"):
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            rec = ModelRecordV4(**data)
            out[rec.model_id] = rec
        return out

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_record(self, record: ModelRecordV4) -> None:
        path = self.registry_dir / f"{record.model_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(record), f, indent=2)


# ---------------------------------------------------------------------------
# Global accessor for ModelRegistryV4 (V4 cockpit)
# ---------------------------------------------------------------------------

_model_registry_global = None


def set_global_model_registry(registry) -> None:
    """
    Register the process-wide ModelRegistry instance.
    """
    global _model_registry_global
    _model_registry_global = registry


def get_global_model_registry():
    """
    Return the process-wide ModelRegistry instance, or None if not set.
    """
    return _model_registry_global
