# mikebot/ui/experience_tab.py

from __future__ import annotations

from typing import Any, Dict

import tkinter as tk
from tkinter import ttk

from mikebot.experience.experience_store_v4 import get_global_experience_store


class ExperienceTab(ttk.Frame):
    """
    View into ExperienceStoreV4:
      - sample counts
      - error counts
      - last N samples
    """

    REFRESH_MS = 1500

    def __init__(self, master: tk.Misc, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)

        self._build_ui()
        self._schedule_refresh()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Label(self, text="Experience Store", font=("Segoe UI", 14, "bold"))
        header.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))

        top = ttk.Frame(self)
        top.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)
        top.rowconfigure(0, weight=1)

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
        add_row(row, "Total Samples", "samples"); row += 1
        add_row(row, "Total Errors", "errors"); row += 1
        add_row(row, "Total Predictions", "predictions"); row += 1
        add_row(row, "Last Sample ID", "last_id"); row += 1

        # Last N samples
        last_frame = ttk.LabelFrame(top, text="Last Samples")
        last_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        last_frame.columnconfigure(0, weight=1)
        last_frame.rowconfigure(0, weight=1)

        self.samples_text = tk.Text(
            last_frame,
            height=12,
            wrap="word",
            state="disabled",
            font=("Consolas", 9),
        )
        self.samples_text.grid(row=0, column=0, sticky="nsew")

        s_vsb = ttk.Scrollbar(last_frame, orient="vertical", command=self.samples_text.yview)
        s_vsb.grid(row=0, column=1, sticky="ns")
        self.samples_text.configure(yscrollcommand=s_vsb.set)

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
        store = get_global_experience_store()
        if store is None:
            return

        # We assume store exposes something like:
        #   get_stats() -> dict
        #   get_last_samples(n: int) -> list
        stats = {}
        last_samples = []

        try:
            get_stats = getattr(store, "get_stats", None)
            if callable(get_stats):
                stats = get_stats() or {}
        except Exception:
            stats = {}

        try:
            get_last = getattr(store, "get_last_samples", None)
            if callable(get_last):
                last_samples = get_last(20) or []
        except Exception:
            last_samples = []

        self._set_label("samples", str(stats.get("samples", 0)))
        self._set_label("errors", str(stats.get("errors", 0)))
        self._set_label("predictions", str(stats.get("predictions", 0)))
        self._set_label("last_id", str(stats.get("last_id", "(none)")))

        self._set_samples(last_samples)

    def _set_label(self, key: str, value: str) -> None:
        lbl = self.labels.get(key)
        if lbl is not None:
            lbl.configure(text=str(value))

    def _set_samples(self, samples: Any) -> None:
        self.samples_text.configure(state="normal")
        self.samples_text.delete("1.0", "end")

        if isinstance(samples, list):
            for s in samples:
                self.samples_text.insert("end", repr(s) + "\n")
        else:
            self.samples_text.insert("end", repr(samples))

        self.samples_text.configure(state="disabled")


def main() -> None:
    root = tk.Tk()
    root.title("Mikebot Studio – Experience Store")
    root.geometry("1000x700")

    style = ttk.Style(root)
    try:
        style.theme_use("vista")
    except Exception:
        pass

    tab = ExperienceTab(root)
    tab.pack(fill="both", expand=True)

    root.mainloop()


if __name__ == "__main__":
    main()