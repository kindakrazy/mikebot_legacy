# mikebot/ui/system_state_tab.py

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import tkinter as tk
from tkinter import ttk

from mikebot.core.training_orchestrator_v4 import get_global_training_orchestrator
from mikebot.core.regime_detector import get_global_regime_detector
from mikebot.minions.personality import get_global_personality_manager
from mikebot.live.orchestrator.config import get_global_live_config

from mikebot.models.model_registry_v4 import get_global_model_registry


class SystemStateTab(ttk.Frame):
    """
    System-level state view:
      - regime
      - personality
      - orchestrator loop status
      - active symbol/model
      - heartbeat / last tick
    """

    REFRESH_MS = 1000

    def __init__(self, master: tk.Misc, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)

        self._build_ui()
        self._schedule_refresh()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Label(self, text="System State", font=("Segoe UI", 14, "bold"))
        header.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))

        info_frame = ttk.Frame(self)
        info_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        info_frame.columnconfigure(1, weight=1)

        self.labels: Dict[str, ttk.Label] = {}

        def add_row(r: int, label: str, key: str) -> None:
            ttk.Label(info_frame, text=label + ":", width=20, anchor="e").grid(
                row=r, column=0, sticky="e", padx=(0, 5), pady=2
            )
            val = ttk.Label(info_frame, text="(unknown)", anchor="w")
            val.grid(row=r, column=1, sticky="w", pady=2)
            self.labels[key] = val

        row = 0
        add_row(row, "Regime", "regime"); row += 1
        add_row(row, "Personality", "personality"); row += 1
        add_row(row, "Orchestrator Status", "orch_status"); row += 1
        add_row(row, "Active Symbol", "symbol"); row += 1
        add_row(row, "Active Model", "model"); row += 1
        add_row(row, "Last Heartbeat", "heartbeat"); row += 1
        add_row(row, "Last Tick", "last_tick"); row += 1
        add_row(row, "Last Error", "last_error"); row += 1

        self.log = tk.Text(
            self,
            height=8,
            wrap="word",
            state="disabled",
            font=("Consolas", 9),
        )
        self.log.grid(row=2, column=0, sticky="nsew", padx=10, pady=(5, 10))
        self.rowconfigure(2, weight=1)

    def _schedule_refresh(self) -> None:
        self.after(self.REFRESH_MS, self._refresh)

    def _refresh(self) -> None:
        try:
            self._update_state()
        except Exception as e:
            self._log(f"Error during refresh: {e}")
        finally:
            self._schedule_refresh()

    def _update_state(self) -> None:
        orch = get_global_training_orchestrator()
        regime = get_global_regime_detector()
        personality = get_global_personality_manager()
        live_cfg = get_global_live_config()
        registry = get_global_model_registry()

        # Regime
        regime_name = getattr(regime, "current_regime", None)
        if callable(regime_name):
            regime_name = regime_name()
        self._set_label("regime", regime_name or "(unknown)")

        # Personality
        persona = getattr(personality, "current_personality", None)
        if callable(persona):
            persona = persona()
        self._set_label("personality", persona or "(unknown)")

        # Orchestrator status
        status = None
        if orch is not None:
            status = getattr(orch, "status", None)
            if callable(status):
                status = status()
        self._set_label("orch_status", status or "(idle)")

        # Active symbol + model
        symbol = None
        if live_cfg is not None:
            symbol = getattr(live_cfg, "active_symbol", None)
            if callable(symbol):
                symbol = symbol()
        self._set_label("symbol", symbol or "(none)")

        model = None
        if registry is not None and symbol:
            try:
                model = registry.get_active_version(symbol)
            except Exception:
                model = None
        self._set_label("model", model or "(none)")

        # Heartbeat / last tick
        heartbeat = None
        last_tick = None
        last_error = None

        if orch is not None:
            heartbeat = getattr(orch, "last_heartbeat", None)
            last_tick = getattr(orch, "last_tick_ts", None)
            last_error = getattr(orch, "last_error", None)

        self._set_label("heartbeat", self._fmt_ts(heartbeat))
        self._set_label("last_tick", self._fmt_ts(last_tick))
        self._set_label("last_error", str(last_error) if last_error else "(none)")

    def _fmt_ts(self, value: Any) -> str:
        if value is None:
            return "(none)"
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(value).isoformat(sep=" ", timespec="seconds")
            except Exception:
                return str(value)
        return str(value)

    def _set_label(self, key: str, value: str) -> None:
        lbl = self.labels.get(key)
        if lbl is not None:
            lbl.configure(text=str(value))

    def _log(self, msg: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")


def main() -> None:
    root = tk.Tk()
    root.title("Mikebot Studio – System State")
    root.geometry("800x600")

    style = ttk.Style(root)
    try:
        style.theme_use("vista")
    except Exception:
        pass

    tab = SystemStateTab(root)
    tab.pack(fill="both", expand=True)

    root.mainloop()


if __name__ == "__main__":
    main()