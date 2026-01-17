from __future__ import annotations
import tkinter as tk
from tkinter import ttk

from mikebot.config.strategy_config_loader import StrategyConfigLoader
from mikebot.strategies.registry import load_strategies


class StrategyConfigTab:
    """
    UI tab for editing per-strategy configuration parameters.
    Supports floats, ints, and booleans.
    Includes hooks for future optimizer integration and reset-to-defaults.
    """

    def __init__(self, root):
        self.root = root
        self.loader = StrategyConfigLoader()
        self.frame = ttk.Frame(root)
        self.frame.pack(fill="both", expand=True)

        ttk.Label(
            self.frame,
            text="Strategy Parameters",
            font=("Arial", 14, "bold")
        ).pack(pady=10)

        # Buttons for future optimizer integration
        button_row = ttk.Frame(self.frame)
        button_row.pack(fill="x", pady=5)

        self.suggest_btn = ttk.Button(
            button_row,
            text="Suggest Settings (coming soon)",
            command=self.apply_suggestions,
            state="disabled"
        )
        self.suggest_btn.pack(side="left", padx=5)

        self.reset_btn = ttk.Button(
            button_row,
            text="Reset to Defaults",
            command=self.reset_to_defaults
        )
        self.reset_btn.pack(side="left", padx=5)

        self.entries = {}

        # Auto-discover strategies
        strategies = load_strategies()

        for strat_name in strategies.keys():
            cfg = self.loader.get(strat_name)

            # Skip strategies with no config block
            if not cfg:
                continue

            box = ttk.LabelFrame(self.frame, text=strat_name)
            box.pack(fill="x", padx=10, pady=10)

            self.entries[strat_name] = {}

            for key, value in cfg.items():
                row = ttk.Frame(box)
                row.pack(fill="x", pady=3)

                ttk.Label(row, text=key, width=25).pack(side="left")

                # Determine widget type based on value type
                if isinstance(value, bool):
                    var = tk.BooleanVar(value=value)
                    widget = ttk.Checkbutton(row, variable=var)
                    widget.pack(side="left")

                elif isinstance(value, int):
                    var = tk.IntVar(value=value)
                    entry = ttk.Entry(row, textvariable=var, width=10)
                    entry.pack(side="left")

                else:
                    # Default to float
                    var = tk.DoubleVar(value=float(value))
                    entry = ttk.Entry(row, textvariable=var, width=10)
                    entry.pack(side="left")

                self.entries[strat_name][key] = var

        ttk.Button(self.frame, text="Save", command=self.save).pack(pady=10)

    def save(self):
        """
        Collect all values and write them back to strategy_configs.json.
        """
        for strat_name, params in self.entries.items():
            new_values = {}

            for key, var in params.items():
                val = var.get()

                # Normalize types
                if isinstance(var, tk.BooleanVar):
                    new_values[key] = bool(val)
                elif isinstance(var, tk.IntVar):
                    new_values[key] = int(val)
                else:
                    new_values[key] = float(val)

            self.loader.update(strat_name, new_values)

    # ----------------------------------------------------------------------
    # Future optimizer integration hooks
    # ----------------------------------------------------------------------

    def apply_suggestions(self):
        """
        Placeholder for StrategyOptimizer integration.
        Will load suggested configs and update UI fields.
        """
        pass

    def reset_to_defaults(self):
        """
        Reset all strategy configs to their default values.
        """
        defaults = self.loader.get_defaults()

        for strat_name, params in self.entries.items():
            if strat_name not in defaults:
                continue

            for key, var in params.items():
                if key in defaults[strat_name]:
                    var.set(defaults[strat_name][key])

        # Save immediately after reset
        for strat_name, values in defaults.items():
            self.loader.update(strat_name, values)
