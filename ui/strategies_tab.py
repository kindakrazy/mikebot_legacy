# mikebot/ui/strategies_tab.py

from __future__ import annotations
import tkinter as tk
from tkinter import ttk

from mikebot.config.strategy_config import (
    load_strategy_toggles,
    save_strategy_toggles,
)
from mikebot.strategies.registry import load_strategies


class StrategiesTab:
    """
    Standalone-safe UI tab for enabling/disabling strategy modules.
    Works both inside the main Studio UI and as a standalone window.
    """

    def __init__(self, root, toggles: dict):
        self.root = root
        self.toggles = toggles

        self.frame = ttk.Frame(root)
        self.frame.pack(fill="both", expand=True)

        ttk.Label(
            self.frame,
            text="Strategy Toggles",
            font=("Arial", 14, "bold")
        ).pack(pady=10)

        self.check_vars = {}

        # Auto-discover strategies
        strategies = load_strategies()

        for name in strategies.keys():
            var = tk.BooleanVar(value=self.toggles.get(name, True))
            chk = ttk.Checkbutton(
                self.frame,
                text=name,
                variable=var,
                command=self._save_toggles
            )
            chk.pack(anchor="w", padx=20, pady=4)
            self.check_vars[name] = var

    def _save_toggles(self):
        for name, var in self.check_vars.items():
            self.toggles[name] = var.get()
        save_strategy_toggles(self.toggles)


# -------------------------------------------------------------------------
# Standalone launcher
# -------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Mikebot Strategy Controls")

    toggles = load_strategy_toggles()

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    from mikebot.ui.strategy_config_tab import StrategyConfigTab

    notebook.add(StrategiesTab(notebook, toggles).frame, text="Strategy Toggles")
    notebook.add(StrategyConfigTab(notebook).frame, text="Strategy Parameters")

    root.mainloop()