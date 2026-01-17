from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Telemetry configuration
# ---------------------------------------------------------------------------

@dataclass
class TelemetryConfig:
    """
    Canonical telemetry configuration.

    Consolidates behavior from:
      - modules/telemetry.py
      - highstrike_diagnostics/system_report.txt
      - highstrike_diagnostics/runtime_errors.txt
      - highstrike_diagnostics/self_heal_log.txt
      - dashboard / dash_monitor
    """

    root: Path
    jsonl_path: Path
    system_report_path: Path
    runtime_errors_path: Path
    self_heal_log_path: Path

    max_jsonl_size_mb: int = 50
    rotate_jsonl: bool = True


# ---------------------------------------------------------------------------
# Telemetry service
# ---------------------------------------------------------------------------

class TelemetryService:
    """
    Canonical mikebot v3 telemetry engine.

    Responsibilities:
      - Emit structured JSONL telemetry events
      - Maintain rolling system reports
      - Log runtime errors
      - Log self-heal events
      - Provide a unified telemetry API for orchestrator + order router
    """

    def __init__(self, config: TelemetryConfig) -> None:
        self.config = config
        self._ensure_paths()

    # ----------------------------------------------------------------------
    # Construction helpers
    # ----------------------------------------------------------------------

    @classmethod
    def from_config(cls, live_cfg) -> TelemetryService:
        root = live_cfg.root

        return cls(
            TelemetryConfig(
                root=root,
                jsonl_path=root / "telemetry" / "events.jsonl",
                system_report_path=root / "telemetry" / "system_report.txt",
                runtime_errors_path=root / "telemetry" / "runtime_errors.txt",
                self_heal_log_path=root / "telemetry" / "self_heal_log.txt",
                max_jsonl_size_mb=live_cfg.global_config.get("telemetry_max_mb", 50),
                rotate_jsonl=live_cfg.global_config.get("telemetry_rotate", True),
            )
        )

    def _ensure_paths(self) -> None:
        self.config.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.system_report_path.parent.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    # Core JSONL emitter
    # ----------------------------------------------------------------------

    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        """
        Emit a structured telemetry event to JSONL.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "payload": payload,
        }

        try:
            self._rotate_if_needed()
            with self.config.jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            log.exception("Telemetry emit failed: %s", exc)

    def _rotate_if_needed(self) -> None:
        if not self.config.rotate_jsonl:
            return

        if not self.config.jsonl_path.exists():
            return

        size_mb = self.config.jsonl_path.stat().st_size / (1024 * 1024)
        if size_mb < self.config.max_jsonl_size_mb:
            return

        # Rotate
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        rotated = self.config.jsonl_path.with_name(f"events_{ts}.jsonl")
        self.config.jsonl_path.rename(rotated)
        log.info("Rotated telemetry JSONL to %s", rotated)

    # ----------------------------------------------------------------------
    # Session lifecycle
    # ----------------------------------------------------------------------

    def emit_session_start(self, session_id: str, started_at: datetime) -> None:
        self._emit(
            "session_start",
            {
                "session_id": session_id,
                "started_at": started_at.isoformat(),
            },
        )

    def emit_session_end(self, session_id: str, ended_at: datetime) -> None:
        self._emit(
            "session_end",
            {
                "session_id": session_id,
                "ended_at": ended_at.isoformat(),
            },
        )

    # ----------------------------------------------------------------------
    # Iteration summary
    # ----------------------------------------------------------------------

    def emit_iteration_summary(
        self,
        session_id: str,
        iteration: int,
        timestamp: datetime,
        minion_decisions: List[Any],
        orders: List[Any],
        health: Dict[str, Any],
    ) -> None:
        """
        Emit a full iteration summary for dashboards and diagnostics.
        """
        self._emit(
            "iteration_summary",
            {
                "session_id": session_id,
                "iteration": iteration,
                "timestamp": timestamp.isoformat(),
                "minion_decisions": [d.to_dict() for d in minion_decisions],
                "orders": [o.to_dict() for o in orders],
                "health": health,
            },
        )

    # ----------------------------------------------------------------------
    # Order events
    # ----------------------------------------------------------------------

    def emit_order_routed(self, order) -> None:
        self._emit(
            "order_routed",
            {
                "symbol": order.symbol,
                "side": order.side.value,
                "lot": order.lot_size,
                "price": order.price,
                "stop_loss": order.stop_loss,
                "take_profit": order.take_profit,
                "comment": order.comment,
            },
        )

    # ----------------------------------------------------------------------
    # Error events
    # ----------------------------------------------------------------------

    def emit_error(self, error_type: str, message: str) -> None:
        """
        Emit structured error telemetry AND append to runtime_errors.txt.
        """
        self._emit(
            "error",
            {
                "type": error_type,
                "message": message,
            },
        )

        try:
            with self.config.runtime_errors_path.open("a", encoding="utf-8") as f:
                f.write(
                    f"{datetime.now(timezone.utc).isoformat()} "
                    f"{error_type}: {message}\n"
                )
        except Exception:
            log.exception("Failed to write runtime error log")

    # ----------------------------------------------------------------------
    # Self-heal events
    # ----------------------------------------------------------------------

    def emit_self_heal(self, message: str) -> None:
        """
        Append to self_heal_log.txt.
        """
        try:
            with self.config.self_heal_log_path.open("a", encoding="utf-8") as f:
                f.write(
                    f"{datetime.now(timezone.utc).isoformat()} {message}\n"
                )
        except Exception:
            log.exception("Failed to write self-heal log")

    # ----------------------------------------------------------------------
    # System report
    # ----------------------------------------------------------------------

    def emit_system_report(self, report: Dict[str, Any]) -> None:
        """
        Write a rolling system report.
        """
        try:
            with self.config.system_report_path.open("w", encoding="utf-8") as f:
                f.write(json.dumps(report, indent=2))
        except Exception:
            log.exception("Failed to write system report")