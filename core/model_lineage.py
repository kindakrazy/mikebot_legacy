# mikebot/core/model_lineage.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import json
import datetime as dt

from mikebot.core.experiment_record import ExperimentRecord


class ModelLineageRegistry:
    """
    Tracks model evolution (lineage) per (symbol, timeframe, model_type).

    Extended to store:
      - strategy_config_used
      - strategy_versions
      - strategy_ids_used
      - strategy_feature_summary
      - strategy_toggles
      - feature_origin

    v4 extension:
      - record_experiment(ExperimentRecord) to support symbol-level MULTITF
        TrainPipeline outputs in a structured way.
      - optional promotion recording for UI / diagnostics.
    """

    def __init__(self, lineage_path: Path) -> None:
        self.lineage_path = lineage_path
        self.data: Dict[str, Any] = self._load()
        # Ensure older files are upgraded to include new fields
        migrated = self._migrate_missing_fields()
        if migrated:
            self._save()

    # ------------------------------------------------------------------
    # INTERNAL LOAD/SAVE
    # ------------------------------------------------------------------

    def _load(self) -> Dict[str, Any]:
        if not self.lineage_path.exists():
            return {}
        try:
            with self.lineage_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _save(self) -> None:
        self.lineage_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lineage_path.open("w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, sort_keys=True)

    # ------------------------------------------------------------------
    # KEY UTILITIES
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(symbol: str, timeframe: str, model_type: str) -> str:
        return f"{symbol}::{timeframe}::{model_type}"

    # ------------------------------------------------------------------
    # VERSION GENERATION
    # ------------------------------------------------------------------

    def _next_version_id(self, key: str) -> str:
        entry = self.data.get(key)
        if not entry or "versions" not in entry or not entry["versions"]:
            return "v1"

        numeric: List[int] = []
        for v in entry["versions"].keys():
            if v.startswith("v"):
                try:
                    numeric.append(int(v[1:]))
                except ValueError:
                    pass

        if not numeric:
            return "v1"

        return f"v{max(numeric) + 1}"

    # ------------------------------------------------------------------
    # PUBLIC API (legacy / existing)
    # ------------------------------------------------------------------

    def add_version(
        self,
        symbol: str,
        timeframe: str,
        model_type: str,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        parent_version: Optional[str] = None,
        regime_performance: Optional[Dict[str, Any]] = None,
        strategy_performance: Optional[Dict[str, Any]] = None,
        strategy_config_used: Optional[Dict[str, Any]] = None,
        strategy_versions: Optional[Dict[str, str]] = None,
        strategy_ids_used: Optional[Dict[str, str]] = None,
        strategy_feature_summary: Optional[Dict[str, Any]] = None,
        strategy_toggles: Optional[Dict[str, Any]] = None,
        feature_origin: Optional[Dict[str, Dict[str, str]]] = None,
        notes: Optional[str] = None,
        set_best: bool = False,
    ) -> str:
        """
        Add a new model version to the lineage, extended with strategy metadata.

        New optional fields:
          - strategy_toggles: snapshot of toggles used when training
          - feature_origin: mapping of namespaced feature -> origin info
        """
        key = self.make_key(symbol, timeframe, model_type)
        entry = self.data.setdefault(key, {"versions": {}, "latest_version": None})

        version_id = self._next_version_id(key)
        now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()

        # Ensure we persist plain Python types where possible
        version_record = {
            "created_at": now,
            "parent": parent_version,
            "config": config or {},
            "metrics": metrics or {},
            "regime_performance": regime_performance or {},
            "strategy_performance": strategy_performance or {},
            # Strategy-aware lineage fields
            "strategy_config_used": strategy_config_used or {},
            "strategy_versions": strategy_versions or {},
            "strategy_ids_used": strategy_ids_used or {},
            "strategy_feature_summary": strategy_feature_summary or {},
            "strategy_toggles": strategy_toggles or {},
            "feature_origin": feature_origin or {},
            "notes": notes or "",
        }

        entry["versions"][version_id] = version_record
        entry["latest_version"] = version_id

        if set_best or "best_version" not in entry:
            entry["best_version"] = version_id

        self.data[key] = entry
        self._save()
        return version_id

    # ------------------------------------------------------------------
    # v4 API: experiment recording
    # ------------------------------------------------------------------

    def record_experiment(
        self,
        record: ExperimentRecord,
        set_best: bool = False,
    ) -> str:
        """
        Record a training experiment as a model version.

        This is the v4-friendly entrypoint that accepts a structured
        ExperimentRecord produced by TrainPipeline.

        Behavior:
          - Uses record.version_id if provided and unique; otherwise falls back
            to _next_version_id.
          - Stores metrics, regime/strategy performance, and strategy-aware
            metadata using the same schema as add_version, keeping JSON
            compatible with existing files.
          - Adds a few non-breaking fields:
              * experiment_type
              * model_tag
              * model_path
              * metrics_path
        """
        symbol = record.symbol
        timeframe = record.timeframe
        model_type = record.model_type

        key = self.make_key(symbol, timeframe, model_type)
        entry = self.data.setdefault(key, {"versions": {}, "latest_version": None})
        versions: Dict[str, Any] = entry.setdefault("versions", {})

        # Decide version_id: respect explicit id if possible, else generate.
        version_id = record.version_id or self._next_version_id(key)
        if version_id in versions:
            # Overwrite semantics if the same version_id is reused.
            # This preserves determinism for idempotent pipelines.
            pass

        now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()

        metrics = record.metrics or {}
        regime_perf = record.regime_performance or metrics.get("regime_performance", {}) or {}
        strat_perf = record.strategy_performance or metrics.get("strategy_performance", {}) or {}

        version_record = {
            "created_at": now,
            "parent": record.parent_version_id,
            "config": {},
            "metrics": metrics,
            "regime_performance": regime_perf,
            "strategy_performance": strat_perf,
            "strategy_config_used": {},
            "strategy_versions": {},
            "strategy_ids_used": {},
            "strategy_feature_summary": {},
            "strategy_toggles": {},
            "feature_origin": {},
            "notes": record.notes or "",
            # v4 extras (non-breaking)
            "experiment_type": record.experiment_type or "",
            "model_tag": record.model_tag or "",
            "model_path": record.model_path or "",
            "metrics_path": record.metrics_path or "",
        }

        versions[version_id] = version_record
        entry["versions"] = versions
        entry["latest_version"] = version_id

        if set_best or "best_version" not in entry:
            entry["best_version"] = version_id

        self.data[key] = entry
        self._save()
        return version_id

    def record_promotion(
        self,
        symbol: str,
        timeframe: str,
        model_type: str,
        version_id: str,
        reason: str = "",
    ) -> None:
        """
        Record a promotion event for a given version. This does not change the
        model registry itself; it is a lineage-side annotation.

        Stored under the lineage entry as:
          entry["promotions"] = [
              {"version_id": ..., "reason": ..., "promoted_at": ...},
              ...
          ]
        """
        key = self.make_key(symbol, timeframe, model_type)
        entry = self.data.setdefault(key, {"versions": {}, "latest_version": None})
        promotions: List[Dict[str, Any]] = entry.setdefault("promotions", [])

        promoted_at = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()
        promotions.append(
            {
                "version_id": version_id,
                "reason": reason,
                "promoted_at": promoted_at,
            }
        )

        entry["promotions"] = promotions
        self.data[key] = entry
        self._save()

    # ------------------------------------------------------------------
    # RETRIEVAL
    # ------------------------------------------------------------------

    def get_latest(
        self,
        symbol: str,
        timeframe: str,
        model_type: str,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        key = self.make_key(symbol, timeframe, model_type)
        entry = self.data.get(key)
        if not entry:
            return None

        latest = entry.get("latest_version")
        if latest is None:
            return None

        record = entry["versions"].get(latest)
        if not record:
            return None

        record = self._ensure_version_fields(record)
        return (latest, record)

    def get_best(
        self,
        symbol: str,
        timeframe: str,
        model_type: str,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        key = self.make_key(symbol, timeframe, model_type)
        entry = self.data.get(key)
        if not entry:
            return None

        best = entry.get("best_version")
        if best is None:
            return None

        record = entry["versions"].get(best)
        if not record:
            return None

        record = self._ensure_version_fields(record)
        return (best, record)

    def get_history(
        self,
        symbol: str,
        timeframe: str,
        model_type: str,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        key = self.make_key(symbol, timeframe, model_type)
        entry = self.data.get(key)
        if not entry or "versions" not in entry:
            return []

        versions = entry["versions"]

        def parse_ts(rec: Dict[str, Any]) -> dt.datetime:
            ts = rec.get("created_at")
            if not ts:
                return dt.datetime.min.replace(tzinfo=dt.timezone.utc)
            try:
                return dt.datetime.fromisoformat(ts)
            except Exception:
                return dt.datetime.min.replace(tzinfo=dt.timezone.utc)

        items = list(versions.items())
        items.sort(key=lambda kv: parse_ts(kv[1]))
        return [(vid, self._ensure_version_fields(rec)) for vid, rec in items]

    # ------------------------------------------------------------------
    # COMPARISON
    # ------------------------------------------------------------------

    def compare_versions(
        self,
        symbol: str,
        timeframe: str,
        model_type: str,
        v1: str,
        v2: str,
        primary_metric: str = "accuracy",
        higher_is_better: bool = True,
    ) -> Optional[str]:
        key = self.make_key(symbol, timeframe, model_type)
        entry = self.data.get(key)
        if not entry or "versions" not in entry:
            return None

        versions = entry["versions"]
        r1 = versions.get(v1)
        r2 = versions.get(v2)
        if not r1 or not r2:
            return None

        m1 = r1.get("metrics", {}).get(primary_metric)
        m2 = r2.get("metrics", {}).get(primary_metric)
        if m1 is None or m2 is None:
            return None

        if higher_is_better:
            return v1 if m1 > m2 else v2
        else:
            return v1 if m1 < m2 else v2

    # ------------------------------------------------------------------
    # UTILITY
    # ------------------------------------------------------------------

    def reload(self) -> None:
        self.data = self._load()
        migrated = self._migrate_missing_fields()
        if migrated:
            self._save()

    def list_keys(self) -> List[str]:
        return sorted(self.data.keys())

    def get_entry(
        self,
        symbol: str,
        timeframe: str,
        model_type: str,
    ) -> Optional[Dict[str, Any]]:
        key = self.make_key(symbol, timeframe, model_type)
        entry = self.data.get(key)
        if not entry:
            return None
        # Do not mutate stored entry here; return a shallow copy with ensured fields
        entry_copy = dict(entry)
        versions = entry_copy.get("versions", {})
        entry_copy["versions"] = {
            vid: self._ensure_version_fields(rec) for vid, rec in versions.items()
        }
        return entry_copy

    # ------------------------------------------------------------------
    # MIGRATION / BACKWARDS COMPATIBILITY HELPERS
    # ------------------------------------------------------------------

    def _ensure_version_fields(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure a version record contains the newer optional fields with safe defaults.
        This does not persist changes to disk; callers that need persistence should call _migrate_missing_fields.
        """
        rec = dict(record)  # shallow copy
        rec.setdefault("strategy_config_used", {})
        rec.setdefault("strategy_versions", {})
        rec.setdefault("strategy_ids_used", {})
        rec.setdefault("strategy_feature_summary", {})
        rec.setdefault("strategy_toggles", {})
        rec.setdefault("feature_origin", {})
        rec.setdefault("notes", "")
        rec.setdefault("regime_performance", {})
        rec.setdefault("strategy_performance", {})
        rec.setdefault("experiment_type", "")
        rec.setdefault("model_tag", "")
        rec.setdefault("model_path", "")
        rec.setdefault("metrics_path", "")
        return rec

    def _migrate_missing_fields(self) -> bool:
        """
        Walk existing data and add missing keys to version records so older files are upgraded.
        Returns True if any changes were made.
        """
        changed = False
        for key, entry in list(self.data.items()):
            versions = entry.get("versions", {})
            for vid, rec in list(versions.items()):
                updated = False

                if "strategy_config_used" not in rec:
                    rec["strategy_config_used"] = {}
                    updated = True
                if "strategy_versions" not in rec:
                    rec["strategy_versions"] = {}
                    updated = True
                if "strategy_ids_used" not in rec:
                    rec["strategy_ids_used"] = {}
                    updated = True
                if "strategy_feature_summary" not in rec:
                    rec["strategy_feature_summary"] = {}
                    updated = True
                if "strategy_toggles" not in rec:
                    rec["strategy_toggles"] = {}
                    updated = True
                if "feature_origin" not in rec:
                    rec["feature_origin"] = {}
                    updated = True
                if "notes" not in rec:
                    rec["notes"] = ""
                    updated = True
                if "regime_performance" not in rec:
                    rec["regime_performance"] = {}
                    updated = True
                if "strategy_performance" not in rec:
                    rec["strategy_performance"] = {}
                    updated = True
                if "experiment_type" not in rec:
                    rec["experiment_type"] = ""
                    updated = True
                if "model_tag" not in rec:
                    rec["model_tag"] = ""
                    updated = True
                if "model_path" not in rec:
                    rec["model_path"] = ""
                    updated = True
                if "metrics_path" not in rec:
                    rec["metrics_path"] = ""
                    updated = True

                if updated:
                    versions[vid] = rec
                    changed = True

            if changed:
                entry["versions"] = versions
                self.data[key] = entry

        return changed