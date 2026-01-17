# mikebot/ui/strategy_engine_tab.py

from __future__ import annotations

from typing import Any, Dict

import tkinter as tk
from tkinter import ttk

from mikebot.strategies.engine_v4 import get_global_strategy_engine


class StrategyEngineTab(ttk.Frame):
    """
    View into StrategyEngineV4:
      - strategies
      - last signal
      - extras
      - runtime context
      - errors
    """

    REFRESH_MS = 1000

    def __init__(self, master: tk.Misc, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)

        self._build_ui()
        self._schedule_refresh()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Label(self, text="Strategy Engine", font=("Segoe UI", 14, "bold"))
        header.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))

        top = ttk.Frame(self)
        top.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)
        top.rowconfigure(1, weight=1)

        # Strategy list
        left = ttk.LabelFrame(top, text="Strategies")
        left.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 5))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(
            left,
            columns=("enabled", "last_signal"),
            show="headings",
            selectmode="browse",
        )
        self.tree.heading("enabled", text="Enabled")
        self.tree.heading("last_signal", text="Last Signal")
        self.tree.column("enabled", width=80, anchor="center")
        self.tree.column("last_signal", width=160, anchor="w")
        self.tree.grid(row=0, column=0, sticky="nsew")

        vsb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=vsb.set)

        # Details
        right = ttk.LabelFrame(top, text="Details")
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        right.columnconfigure(1, weight=1)

        self.labels: Dict[str, ttk.Label] = {}

        def add_row(r: int, label: str, key: str) -> None:
            ttk.Label(right, text=label + ":", width=18, anchor="e").grid(
                row=r, column=0, sticky="e", padx=(0, 5), pady=2
            )
            val = ttk.Label(right, text="(unknown)", anchor="w")
            val.grid(row=r, column=1, sticky="w", pady=2)
            self.labels[key] = val

        row = 0
        add_row(row, "Selected Strategy", "name"); row += 1
        add_row(row, "Enabled", "enabled"); row += 1
        add_row(row, "Last Signal", "last_signal"); row += 1
        add_row(row, "Last Extras", "extras"); row += 1
        add_row(row, "Last Error", "error"); row += 1

        # Context
        ctx_frame = ttk.LabelFrame(top, text="Runtime Context")
        ctx_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 0), pady=(5, 0))
        ctx_frame.columnconfigure(0, weight=1)
        ctx_frame.rowconfigure(0, weight=1)

        self.ctx_text = tk.Text(
            ctx_frame,
            height=8,
            wrap="word",
            state="disabled",
            font=("Consolas", 9),
        )
        self.ctx_text.grid(row=0, column=0, sticky="nsew")

        ctx_vsb = ttk.Scrollbar(ctx_frame, orient="vertical", command=self.ctx_text.yview)
        ctx_vsb.grid(row=0, column=1, sticky="ns")
        self.ctx_text.configure(yscrollcommand=ctx_vsb.set)

        self.tree.bind("<<TreeviewSelect>>", self._on_select)

    def _schedule_refresh(self) -> None:
        self.after(self.REFRESH_MS, self._refresh)

    def _refresh(self) -> None:
        try:
            self._update_strategies()
        except Exception:
            # Keep UI alive even if engine misbehaves
            pass
        finally:
            self._schedule_refresh()

    def _update_strategies(self) -> None:
        engine = get_global_strategy_engine()
        self.tree.delete(*self.tree.get_children())

        if engine is None:
            return

        # We assume engine exposes something like: engine.list_strategies()
        # Each item: {"name": ..., "enabled": bool, "last_signal": ...}
        try:
            strategies = getattr(engine, "list_strategies", None)
            if callable(strategies):
                items = strategies()
            else:
                items = []
        except Exception:
            items = []

        for strat in items or []:
            name = strat.get("name", "")
            enabled = "yes" if strat.get("enabled", True) else "no"
            last_signal = str(strat.get("last_signal", ""))
            self.tree.insert("", "end", iid=name, values=(enabled, last_signal))

    def _on_select(self, event: Any) -> None:
        engine = get_global_strategy_engine()
        if engine is None:
            return

        sel = self.tree.selection()
        if not sel:
            return

        name = sel[0]

        # We assume engine exposes something like: engine.get_strategy_state(name)
        try:
            get_state = getattr(engine, "get_strategy_state", None)
            state = get_state(name) if callable(get_state) else None
        except Exception:
            state = None

        state = state or {}

        self._set_label("name", name)
        self._set_label("enabled", "yes" if state.get("enabled", True) else "no")
        self._set_label("last_signal", str(state.get("last_signal", "")))
        self._set_label("extras", str(state.get("extras", "")))
        self._set_label("error", str(state.get("error", "")))

        ctx = state.get("context", {})
        self._set_ctx(ctx)

    def _set_label(self, key: str, value: str) -> None:
        lbl = self.labels.get(key)
        if lbl is not None:
            lbl.configure(text=str(value))

    def _set_ctx(self, ctx: Any) -> None:
        self.ctx_text.configure(state="normal")
        self.ctx_text.delete("1.0", "end")
        self.ctx_text.insert("1.0", repr(ctx))
        self.ctx_text.configure(state="disabled")


def main() -> None:
    root = tk.Tk()
    root.title("Mikebot Studio – Strategy Engine")
    root.geometry("1000x700")

    style = ttk.Style(root)
    try:
        style.theme_use("vista")
    except Exception:
        pass

    tab = StrategyEngineTab(root)
    tab.pack(fill="both", expand=True)

    root.mainloop()


if __name__ == "__main__":
    main()