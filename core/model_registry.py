from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone


# ======================================================================
# Core load/save
# ======================================================================

def load_registry(path: Path) -> Dict[str, Any]:
    """
    Load a JSON model registry from disk. If missing, return a minimal registry.
    Backward-compatible with older TF-based registries.
    """
    if not path.exists():
        return {
            "schema_version": "mikebot-model-registry-2.0",
            "models": {},
            "active_models": {},
        }

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {
            "schema_version": "mikebot-model-registry-2.0",
            "models": {},
            "active_models": {},
        }

    # Backfill missing fields
    data.setdefault("models", {})
    data.setdefault("active_models", {})
    data.setdefault("schema_version", "mikebot-model-registry-2.0")

    return data


def save_registry(path: Path, registry: Dict[str, Any]) -> None:
    """
    Persist the registry to disk atomically (best-effort).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, sort_keys=True)
    tmp.replace(path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ======================================================================
# Update entry
# ======================================================================

def update_entry(
    registry: Dict[str, Any],
    key: str,
    model_path: str,
    metrics: Dict[str, Any],
    promoted: bool = False,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Update or create a registry entry for a model key.

    Key format (new architecture):
        SYMBOL::MULTITF::MODELTYPE

    Backward-compatible with older TF-based keys.
    """
    entry = registry.setdefault("models", {}).get(key, {})
    lineage = entry.get("lineage", {})

    new_entry = {
        "model_path": model_path,
        "created_at": _now_iso(),
        "metrics": metrics,
        "status": "active" if promoted else "staging",
        "lineage": {
            "parent": lineage.get("parent"),
            "notes": notes or lineage.get("notes"),
        },
    }

    registry["models"][key] = new_entry
    return new_entry


# ======================================================================
# ModelRegistry wrapper
# ======================================================================

@dataclass
class ModelRegistry:
    """
    Symbol-level model registry.

    New architecture:
      - One model per symbol
      - Scope is always "MULTITF"
      - Keys are: SYMBOL::MULTITF::MODELTYPE

    Backward-compatible with older TF-based registries.
    """

    path: Path

    def __post_init__(self) -> None:
        self._data: Dict[str, Any] = load_registry(self.path)

    # --------------------------------------------------------------
    # Core IO
    # --------------------------------------------------------------

    def reload(self) -> None:
        self._data = load_registry(self.path)

    def save(self) -> None:
        save_registry(self.path, self._data)

    @property
    def data(self) -> Dict[str, Any]:
        return self._data

    # --------------------------------------------------------------
    # Key helpers
    # --------------------------------------------------------------

    @staticmethod
    def make_key(symbol: str, model_type: str) -> str:
        """
        Build a stable key for symbol-level MULTITF models.

        Format:
            SYMBOL::MULTITF::MODELTYPE
        """
        return f"{symbol}::MULTITF::{model_type}"

    # --------------------------------------------------------------
    # Introspection
    # --------------------------------------------------------------

    def list_symbols(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Returns a structure describing available models:

            {
                "BTCUSD": {
                    "MULTITF": {
                        "rf": True,
                        "xgb": True
                    }
                }
            }

        Backward-compatible: older TF-based models are still shown.
        """
        result: Dict[str, Dict[str, Dict[str, Any]]] = {}

        # 1) Active models (authoritative)
        active = self._data.get("active_models", {})
        for symbol, scopes in active.items():
            for scope, types in scopes.items():
                for mtype in types.keys():
                    result.setdefault(symbol, {}).setdefault(scope, {})[mtype] = True

        # 2) All models (parse keys)
        models = self._data.get("models", {})
        for key in models.keys():
            if "::" not in key:
                continue

            parts = key.split("::")
            if len(parts) != 3:
                continue

            symbol, scope, mtype = parts
            scope = scope.upper()

            result.setdefault(symbol, {}).setdefault(scope, {})[mtype] = True

        return result

    def get_symbol_list(self) -> List[str]:
        return sorted(self.list_symbols().keys())

    def get_scopes_for_symbol(self, symbol: str) -> List[str]:
        """
        Returns scopes for a symbol.

        New architecture: always ["MULTITF"] for new models.
        """
        scopes = self.list_symbols().get(symbol, {})
        return sorted(scopes.keys())

    def get_types_for_symbol_scope(self, symbol: str, scope: str) -> List[str]:
        return sorted(self.list_symbols().get(symbol, {}).get(scope, {}).keys())

    # --------------------------------------------------------------
    # Active model handling (symbol-level)
    # --------------------------------------------------------------

    def update_active_version(
        self,
        symbol: str,
        model_type: str,
        version_id: str,
    ) -> None:
        """
        Mark a specific version as active for (symbol, MULTITF, model_type).

        Writes to:
            active_models[symbol]["MULTITF"][model_type] = version_id
        """
        am = self._data.setdefault("active_models", {})
        am.setdefault(symbol, {}).setdefault("MULTITF", {})[model_type] = version_id
        self.save()

        # Also update status in models
        models = self._data.setdefault("models", {})
        if version_id in models:
            models[version_id]["status"] = "active"
            self.save()

    def get_active_version(
        self,
        symbol: str,
        model_type: str,
    ) -> Optional[str]:
        """
        Returns the active version for (symbol, MULTITF, model_type).
        """
        am = self._data.get("active_models", {})
        return (
            am.get(symbol, {})
            .get("MULTITF", {})
            .get(model_type)
        )

    # --------------------------------------------------------------
    # REQUIRED BY LEARNER: get_active_model_type
    # --------------------------------------------------------------

    def get_active_model_type(
        self,
        symbol: str,
        timeframe: str = "MULTITF",
    ) -> Optional[str]:
        """
        Return the active model type for a symbol.

        LearnerService calls this to determine which model family
        (rf, xgb, etc.) should be retrained.

        We ignore timeframe because the new architecture is symbol-level.
        """
        active = self._data.get("active_models", {}).get(symbol, {})
        scope = active.get("MULTITF", {})

        if not scope:
            return None

        # Return the first active model type
        return next(iter(scope.keys()))

    # --------------------------------------------------------------
    # REQUIRED BY METATRAINER: register_model
    # --------------------------------------------------------------

    def register_model(
        self,
        symbol: str,
        timeframe: str,
        model_type: str,
        version: str,
        model_path: Path,
        metrics_path: Path,
        metrics_summary: Dict[str, Any],
        set_current: bool = True,
        set_best: bool = True,
        notes: Optional[str] = None,
    ) -> None:
        """
        Register a model version in the registry.

        This is the preferred API for MetaTrainer.
        """
        key = self.make_key(symbol, model_type)

        entry = {
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
            "metrics": metrics_summary,
            "created_at": _now_iso(),
            "status": "active" if set_current else "staging",
            "notes": notes,
        }

        self._data.setdefault("models", {})[key] = entry

        if set_current:
            self.update_active_version(symbol, model_type, version)

        self.save()