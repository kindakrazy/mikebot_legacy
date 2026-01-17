# mikebot/diagnostics/diagnostics_v4.py

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd


@dataclass
class DiagnosticEventV4:
    """
    A single diagnostic event.

    This is intentionally generic and can represent:
    - model predictions
    - strategy signals
    - execution decisions
    - order routing results
    - training metrics
    - arbitrary metadata snapshots
    """
    timestamp: Any
    event_type: str
    payload: Dict[str, Any]


class DiagnosticsV4:
    """
    Unified diagnostics and lineage logger for v4.

    Responsibilities:
    - Append structured diagnostic events
    - Persist them to disk as JSONL
    - Load them back into DataFrames for analysis
    - Provide lineage tagging for models, strategies, and experiments

    This module is storage-agnostic and does not depend on any v3 logging.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        self.events_path = self.root / "events.jsonl"
        self.lineage_path = self.root / "lineage.json"

        # In-memory buffer for fast appends
        self._buffer: list[DiagnosticEventV4] = []

    # ------------------------------------------------------------------
    # Event logging
    # ------------------------------------------------------------------

    def log_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        timestamp: Optional[Any] = None,
    ) -> None:
        """
        Append a diagnostic event to the buffer.
        """
        ts = timestamp if timestamp is not None else pd.Timestamp.utcnow().isoformat()
        evt = DiagnosticEventV4(timestamp=ts, event_type=event_type, payload=payload)
        self._buffer.append(evt)

    def flush(self) -> None:
        """
        Persist buffered events to disk as JSONL.
        """
        if not self._buffer:
            return

        with open(self.events_path, "a", encoding="utf-8") as f:
            for evt in self._buffer:
                f.write(json.dumps(asdict(evt)) + "\n")

        self._buffer.clear()

    # ------------------------------------------------------------------
    # Lineage tracking
    # ------------------------------------------------------------------

    def set_lineage(self, lineage: Dict[str, Any]) -> None:
        """
        Overwrite lineage metadata.
        """
        with open(self.lineage_path, "w", encoding="utf-8") as f:
            json.dump(lineage, f, indent=2)

    def load_lineage(self) -> Dict[str, Any]:
        """
        Load lineage metadata.
        """
        if not self.lineage_path.exists():
            return {}
        with open(self.lineage_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Loading and analysis
    # ------------------------------------------------------------------

    def load_events(self) -> pd.DataFrame:
        """
        Load all diagnostic events into a DataFrame.

        Columns:
            timestamp
            event_type
            payload (dict)
        """
        if not self.events_path.exists():
            return pd.DataFrame()

        rows = []
        with open(self.events_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    rows.append(obj)
                except Exception:
                    continue

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)

    def load_events_by_type(self, event_type: str) -> pd.DataFrame:
        """
        Load only events of a specific type.
        """
        df = self.load_events()
        if df.empty:
            return df
        return df[df["event_type"] == event_type].reset_index(drop=True)