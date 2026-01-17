# mikebot/ui/model_learning_tab.py

from __future__ import annotations

from typing import Any, Dict

import tkinter as tk
from tkinter import ttk

from mikebot.live.services.learner import get_global_learner_service
from mikebot.models.model_registry_v4 import get_global_model_registry


class ModelLearningTab(ttk.Frame):
    """
    View into model learning:
      - active model
      - drift
      - retrain history
      - sample counts
    """

    REFRESH_MS = 1500

    def __init__(self, master: tk.Misc, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)

        self._build_ui()
        self._schedule_refresh()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Label(self, text="Model Learning", font=("Segoe UI", 14, "bold"))
        header.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))

        top = ttk.Frame(self)
        top.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)
        top.rowconfigure(1, weight=1)

        # Summary
        summary = ttk.LabelFrame(top, text="Summary")
        summary.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        summary.columnconfigure(1, weight=1)

        self.labels: Dict[str, ttk.Label] = {}

        def add_row(r: int, label: str, key: str) -> None:
            ttk.Label(summary, text=label + ":", width=18, anchor="e").grid(
                row=r, column=0, sticky="e", padx=(0, 5), pady=2
            )
            val = ttk.Label(summary, text="(unknown)", anchor="w")
            val.grid(row=r, column=1, sticky="w", pady=2)
            self.labels[key] = val

        row = 0
        add_row(row, "Active Symbol", "symbol"); row += 1
        add_row(row, "Active Model", "model"); row += 1
        add_row(row, "Drift Score", "drift"); row += 1
        add_row(row, "Total Samples", "samples"); row += 1
        add_row(row, "Last Retrain", "last_retrain"); row += 1
        add_row(row, "Last Error", "last_error"); row += 1

        # History
        history_frame = ttk.LabelFrame(top, text="Retrain History")
        history_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(5, 0))
        history_frame.columnconfigure(0, weight=1)
        history_frame.rowconfigure(0, weight=1)

        self.history_text = tk.Text(
            history_frame,
            height=10,
            wrap="word",
            state="disabled",
            font=("Consolas", 9),
        )
        self.history_text.grid(row=0, column=0, sticky="nsew")

        h_vsb = ttk.Scrollbar(history_frame, orient="vertical", command=self.history_text.yview)
        h_vsb.grid(row=0, column=1, sticky="ns")
        self.history_text.configure(yscrollcommand=h_vsb.set)

    def _schedule_refresh(self) -> None:
        self.after(self.REFRESH_MS, self._refresh)

    def _refresh(self) -> None:
        try:
            self._update_state()
        except Exception:
            pass
        finally:
            self._schedule_refresh()

    def _update_state(self) -> None:
        learner = get_global_learner_service()
        registry = get_global_model_registry()

        symbol = None
        model = None
        drift = None
        samples = None
        last_retrain = None
        last_error = None
        history = None

        if learner is not None:
            # We assume learner exposes these attributes or methods.
            symbol = getattr(learner, "active_symbol", None)
            if callable(symbol):
                symbol = symbol()

            drift = getattr(learner, "current_drift", None)
            if callable(drift):
                drift = drift()

            samples = getattr(learner, "total_samples", None)
            if callable(samples):
                samples = samples()

            last_retrain = getattr(learner, "last_retrain_ts", None)
            if callable(last_retrain):
                last_retrain = last_retrain()

            last_error = getattr(learner, "last_error", None)
            if callable(last_error):
                last_error = last_error()

            history = getattr(learner, "get_retrain_history", None)
            if callable(history):
                history = history()
            else:
                history = None

        if registry is not None and symbol:
            try:
                model = registry.get_active_version(symbol)
            except Exception:
                model = None

        self._set_label("symbol", symbol or "(none)")
        self._set_label("model", model or "(none)")
        self._set_label("drift", f"{drift:.6f}" if isinstance(drift, (int, float)) else str(drift or "(none)"))
        self._set_label("samples", str(samples or 0))
        self._set_label("last_retrain", str(last_retrain or "(none)"))
        self._set_label("last_error", str(last_error or "(none)"))

        self._set_history(history or [])

    def _set_label(self, key: str, value: str) -> None:
        lbl = self.labels.get(key)
        if lbl is not None:
            lbl.configure(text=str(value))

    def _set_history(self, history: Any) -> None:
        self.history_text.configure(state="normal")
        self.history_text.delete("1.0", "end")

        if isinstance(history, list):
            for item in history:
                self.history_text.insert("end", repr(item) + "\n")
        else:
            self.history_text.insert("end", repr(history))

        self.history_text.configure(state="disabled")


def main() -> None:
    root = tk.Tk()
    root.title("Mikebot Studio – Model Learning")
    root.geometry("900x600")

    style = ttk.Style(root)
    try:
        style.theme_use("vista")
    except Exception:
        pass

    tab = ModelLearningTab(root)
    tab.pack(fill="both", expand=True)

    root.mainloop()


if __name__ == "__main__":
    main()