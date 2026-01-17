"""
Data Ingestion & Training tab for Mikebot Studio (unified-v3.5).

Standalone Tkinter application that will later be embedded as a tab inside
mikebot/ui/studio.py. It provides:

    - CSV file selection (ingestion)
    - Raw data preview (pandas -> Treeview)
    - Normalization via: python -m mikebot.data.normalize_csv <file>
    - Normalized data preview loaded from mikebot/data/normalized/<SYMBOL+TF>/*.normalized.csv
    - Training via: python -m mikebot.core.train_pipeline
    - Metadata panel for basic data health
    - Log panel for all operations

Directory conventions:

    - ROOT = repo root (two levels up from this file)
    - ROOT/uploads             : raw uploaded CSV files
    - ROOT/data/raw            : raw CSV files
    - ROOT/data/normalized     : normalized CSVs in per-symbol+tf folders
      (normalized files follow: <SYMBOLTF>.<YYYYMMDD-YYYYMMDD>.normalized.csv)
    - python -m mikebot.data.normalize_csv <file>
    - python -m mikebot.core.train_pipeline

All paths in comments use forward slashes to avoid escape issues on Windows.
"""

from __future__ import annotations

import subprocess
import traceback
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd
import re


# ---------------------------------------------------------------------------
# Paths and helpers
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve()
# mikebot/ui/data_tab.py -> mikebot/
ROOT = HERE.parent.parent
UPLOADS = ROOT / "uploads"
DATA_RAW = ROOT / "data" / "raw"
DATA_NORMALIZED = ROOT / "data" / "normalized"
# We assume the venv Python is at venv/Scripts/python.exe relative to repo root.
# If you use a different venv layout, adjust this path.
VENV_PY = ROOT / "venv" / "Scripts" / "python.exe"


def ensure_dirs() -> None:
    for p in (UPLOADS, DATA_RAW, DATA_NORMALIZED):
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Silent failure is acceptable here; errors will surface later on use.
            pass


def ts() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")


def detect_symbol_from_filename(fname: str) -> str:
    """
    Heuristic: use the first token before '_' as symbol, uppercased.

    Model 1 (symbol+timeframe as one identity):

        'BTCUSD1.csv'              -> 'BTCUSD1'
        'LTCUSD15.csv'             -> 'LTCUSD15'
        'BTCUSD1_M1_202401.csv'    -> 'BTCUSD1'
    """
    stem = Path(fname).stem
    parts = stem.split("_")
    if not parts:
        return stem.upper()
    return parts[0].upper()


def detect_timeframe_from_symbol(symbol: str) -> str:
    """
    Extract trailing digits + optional letter as timeframe.

        'BTCUSD1'   -> '1'
        'LTCUSD15'  -> '15'
        'BTCUSDM1'  -> 'M1'
        'ETHUSDH1'  -> 'H1'

    Falls back to 'M15' if no timeframe pattern is found.
    """
    m = re.search(r"([0-9]+[A-Z]?)$", symbol.upper())
    return m.group(1) if m else "M15"


def find_normalized_file_for_symbol(symbol: str) -> Optional[Path]:
    """
    Legacy helper (not used in the main flow anymore, but kept for compatibility).

    Previously looked in DATA_RAW for files that:
        - contain the symbol (case-insensitive), and
        - contain the word 'normalized' in the stem.

    Now, normalized files live under:
        DATA_NORMALIZED/<SYMBOLTF>/*.normalized.csv
    """
    # Kept for reference; main flow now uses DATA_NORMALIZED/<symbol> instead.
    if not DATA_RAW.exists():
        return None

    symbol_u = symbol.upper()
    candidates: List[Tuple[float, Path]] = []

    for p in DATA_RAW.glob("*.csv"):
        stem_u = p.stem.upper()
        if symbol_u in stem_u and "NORMALIZED" in stem_u:
            try:
                mtime = p.stat().st_mtime
                candidates.append((mtime, p))
            except Exception:
                continue

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# ---------------------------------------------------------------------------
# DataTab widget
# ---------------------------------------------------------------------------


class DataTab(ttk.Frame):
    """
    Data Ingestion & Training tab for Mikebot Studio.

    Studio-style layout:
        - Left: ingestion + actions
        - Center: raw + normalized previews
        - Right: metadata
        - Bottom: log panel
    """

    def __init__(self, master: tk.Misc, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)

        ensure_dirs()

        # State
        self.selected_file: Optional[Path] = None
        self.selected_symbol: str = ""          # Model 1: symbol+timeframe identity (e.g., BTCUSD1)
        self.selected_timeframe: Optional[str] = None
        self.raw_df: Optional[pd.DataFrame] = None
        self.norm_df: Optional[pd.DataFrame] = None
        self.normalized_path: Optional[Path] = None

        self.auto_train_var = tk.BooleanVar(value=False)

        self._build_ui()

    # ------------------------------------------------------------------ #
    # UI construction                                                    #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        # Grid: left / center / right | bottom log
        self.columnconfigure(0, weight=0)  # left
        self.columnconfigure(1, weight=3)  # center
        self.columnconfigure(2, weight=1)  # right
        self.rowconfigure(0, weight=1)     # main
        self.rowconfigure(1, weight=0)     # log

        # Left: ingestion + actions
        left = ttk.Frame(self)
        left.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        left.columnconfigure(0, weight=1)
        self._build_left_panel(left)

        # Center: raw + normalized previews
        center = ttk.Frame(self)
        center.grid(row=0, column=1, sticky="nsew", padx=4, pady=8)
        center.rowconfigure(1, weight=1)
        center.rowconfigure(3, weight=1)
        center.columnconfigure(0, weight=1)
        self._build_center_panel(center)

        # Right: metadata
        right = ttk.Frame(self)
        right.grid(row=0, column=2, sticky="nsew", padx=(4, 8), pady=8)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        self._build_right_panel(right)

        # Bottom: log panel
        bottom = ttk.LabelFrame(self, text="Log")
        bottom.grid(row=1, column=0, columnspan=3, sticky="ew", padx=8, pady=(0, 8))
        bottom.columnconfigure(0, weight=1)
        self._build_log_panel(bottom)

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        # Ingestion frame
        ingest = ttk.LabelFrame(parent, text="File Ingestion")
        ingest.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ingest.columnconfigure(0, weight=1)

        btn_select = ttk.Button(
            ingest,
            text="Select CSV file…",
            command=self._on_select_file,
        )
        btn_select.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 2))

        self.selected_file_label = ttk.Label(
            ingest,
            text="Selected: (none)",
            wraplength=220,
        )
        self.selected_file_label.grid(row=1, column=0, sticky="w", padx=4, pady=2)

        self.detected_symbol_label = ttk.Label(
            ingest,
            text="Detected symbol: (none)",
            foreground="gray",
        )
        self.detected_symbol_label.grid(row=2, column=0, sticky="w", padx=4, pady=(0, 4))

        btn_ingest = ttk.Button(
            ingest,
            text="Ingest file (copy to /uploads)",
            command=self._on_ingest_file,
        )
        btn_ingest.grid(row=3, column=0, sticky="ew", padx=4, pady=(2, 4))

        # Actions frame
        actions = ttk.LabelFrame(parent, text="Actions")
        actions.grid(row=1, column=0, sticky="ew")
        actions.columnconfigure(0, weight=1)

        btn_normalize = ttk.Button(
            actions,
            text="Normalize data",
            command=self._on_normalize,
        )
        btn_normalize.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 2))

        btn_refresh_norm = ttk.Button(
            actions,
            text="Refresh normalized preview",
            command=self._on_refresh_normalized_preview,
        )
        btn_refresh_norm.grid(row=1, column=0, sticky="ew", padx=4, pady=2)

        btn_train = ttk.Button(
            actions,
            text="Train model (core.train_pipeline)",
            command=self._on_train,
        )
        btn_train.grid(row=2, column=0, sticky="ew", padx=4, pady=(2, 4))

        auto_train_chk = ttk.Checkbutton(
            actions,
            text="Auto-train after normalize",
            variable=self.auto_train_var,
        )
        auto_train_chk.grid(row=3, column=0, sticky="w", padx=4, pady=(0, 4))

    def _build_center_panel(self, parent: ttk.Frame) -> None:
        # Raw Preview
        raw_label = ttk.Label(parent, text="Raw Data Preview", font=("Segoe UI", 10, "bold"))
        raw_label.grid(row=0, column=0, sticky="w")

        raw_frame = ttk.Frame(parent)
        raw_frame.grid(row=1, column=0, sticky="nsew", pady=(4, 8))
        raw_frame.columnconfigure(0, weight=1)
        raw_frame.rowconfigure(0, weight=1)

        self.raw_tree = ttk.Treeview(raw_frame, columns=(), show="headings")
        self.raw_tree.grid(row=0, column=0, sticky="nsew")

        raw_vsb = ttk.Scrollbar(raw_frame, orient="vertical", command=self.raw_tree.yview)
        raw_vsb.grid(row=0, column=1, sticky="ns")
        self.raw_tree.configure(yscrollcommand=raw_vsb.set)

        raw_hsb = ttk.Scrollbar(raw_frame, orient="horizontal", command=self.raw_tree.xview)
        raw_hsb.grid(row=1, column=0, sticky="ew")
        self.raw_tree.configure(xscrollcommand=raw_hsb.set)

        # Normalized Preview
        norm_label = ttk.Label(parent, text="Normalized Data Preview", font=("Segoe UI", 10, "bold"))
        norm_label.grid(row=2, column=0, sticky="w", pady=(4, 0))

        norm_frame = ttk.Frame(parent)
        norm_frame.grid(row=3, column=0, sticky="nsew", pady=(4, 0))
        norm_frame.columnconfigure(0, weight=1)
        norm_frame.rowconfigure(0, weight=1)

        self.norm_tree = ttk.Treeview(norm_frame, columns=(), show="headings")
        self.norm_tree.grid(row=0, column=0, sticky="nsew")

        norm_vsb = ttk.Scrollbar(norm_frame, orient="vertical", command=self.norm_tree.yview)
        norm_vsb.grid(row=0, column=1, sticky="ns")
        self.norm_tree.configure(yscrollcommand=norm_vsb.set)

        norm_hsb = ttk.Scrollbar(norm_frame, orient="horizontal", command=self.norm_tree.xview)
        norm_hsb.grid(row=1, column=0, sticky="ew")
        self.norm_tree.configure(xscrollcommand=norm_hsb.set)

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Metadata", font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, sticky="w"
        )

        meta = ttk.Frame(parent)
        meta.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        meta.columnconfigure(1, weight=1)

        self.meta_file = ttk.Label(meta, text="File: (none)", wraplength=260)
        self.meta_file.grid(row=0, column=0, columnspan=2, sticky="w", padx=4, pady=2)

        self.meta_symbol = ttk.Label(meta, text="Symbol: (none)")
        self.meta_symbol.grid(row=1, column=0, columnspan=2, sticky="w", padx=4, pady=2)

        self.meta_rows = ttk.Label(meta, text="Rows: -")
        self.meta_rows.grid(row=2, column=0, columnspan=2, sticky="w", padx=4, pady=2)

        self.meta_cols = ttk.Label(meta, text="Columns: -")
        self.meta_cols.grid(row=3, column=0, columnspan=2, sticky="w", padx=4, pady=2)

        self.meta_time = ttk.Label(meta, text="Time range: -")
        self.meta_time.grid(row=4, column=0, columnspan=2, sticky="w", padx=4, pady=2)

        self.meta_missing = ttk.Label(meta, text="Missing values: -")
        self.meta_missing.grid(row=5, column=0, columnspan=2, sticky="w", padx=4, pady=2)

        self.meta_dupes = ttk.Label(meta, text="Duplicates (normalized): -")
        self.meta_dupes.grid(row=6, column=0, columnspan=2, sticky="w", padx=4, pady=2)

    def _build_log_panel(self, parent: ttk.Frame) -> None:
        self.log_text = tk.Text(
            parent,
            height=6,
            wrap="word",
            state="disabled",
            font=("Consolas", 9),
        )
        self.log_text.grid(row=0, column=0, sticky="ew")

        log_scroll = ttk.Scrollbar(parent, orient="vertical", command=self.log_text.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scroll.set)

    # ------------------------------------------------------------------ #
    # Logging helpers                                                    #
    # ------------------------------------------------------------------ #

    def _log(self, msg: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{ts()}] {msg}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    # ------------------------------------------------------------------ #
    # File selection & ingestion                                         #
    # ------------------------------------------------------------------ #

    def _on_select_file(self) -> None:
        fname = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        )
        if not fname:
            return

        path = Path(fname)
        self.selected_file = path
        self.selected_file_label.configure(text=f"Selected: {path.name}")

        sym = detect_symbol_from_filename(path.name)
        self.selected_symbol = sym
        self.selected_timeframe = detect_timeframe_from_symbol(sym)

        self.detected_symbol_label.configure(
            text=f"Detected symbol: {sym} (tf={self.selected_timeframe})",
            foreground="black",
        )

        self._log(f"Selected file: {path}")
        self._load_raw_df(path)

    def _on_ingest_file(self) -> None:
        if not self.selected_file:
            messagebox.showinfo("No file", "Please select a CSV file first.")
            return

        try:
            dest = UPLOADS / self.selected_file.name
            dest.write_bytes(self.selected_file.read_bytes())
            self._log(f"Ingested file -> {dest}")
        except Exception as e:
            traceback_str = traceback.format_exc()
            messagebox.showerror("Error ingesting file", str(e))
            self._log(f"Error ingesting file: {e}\n{traceback_str}")

    # ------------------------------------------------------------------ #
    # Raw preview & metadata                                             #
    # ------------------------------------------------------------------ #

    def _load_raw_df(self, path: Path) -> None:
        try:
            df = pd.read_csv(path)
            self.raw_df = df
            self._log(f"Loaded raw data: {len(df)} rows, {len(df.columns)} columns")
            self._update_raw_tree(df)
            self._update_metadata_from_raw(df, path)
        except Exception as e:
            traceback_str = traceback.format_exc()
            messagebox.showerror("Error loading CSV", str(e))
            self._log(f"Error loading CSV: {e}\n{traceback_str}")

    def _update_raw_tree(self, df: pd.DataFrame, max_rows: int = 200) -> None:
        self._populate_tree(self.raw_tree, df, max_rows)

    def _update_metadata_from_raw(self, df: pd.DataFrame, path: Path) -> None:
        self.meta_file.configure(text=f"File: {path.name}")
        if self.selected_symbol:
            self.meta_symbol.configure(
                text=f"Symbol: {self.selected_symbol} (tf={self.selected_timeframe or 'M15'})"
            )
        else:
            self.meta_symbol.configure(text="Symbol: (none)")

        rows, cols = df.shape
        self.meta_rows.configure(text=f"Rows: {rows}")
        self.meta_cols.configure(text=f"Columns: {cols}")

        # Time range if ts_utc column exists
        if "ts_utc" in df.columns:
            try:
                ts_series = pd.to_datetime(df["ts_utc"], errors="coerce")
                tmin = ts_series.min()
                tmax = ts_series.max()
                if pd.isna(tmin) or pd.isna(tmax):
                    self.meta_time.configure(text="Time range: (unparseable)")
                else:
                    self.meta_time.configure(text=f"Time range: {tmin} → {tmax}")
            except Exception:
                self.meta_time.configure(text="Time range: (error)")
        else:
            self.meta_time.configure(text="Time range: (no ts_utc column)")

        # Missing values (overall percentage)
        try:
            total_cells = df.shape[0] * max(1, df.shape[1])
            missing_cells = int(df.isna().sum().sum())
            pct = (missing_cells / total_cells) * 100.0 if total_cells else 0.0
            self.meta_missing.configure(
                text=f"Missing values: {missing_cells} ({pct:.2f}%)"
            )
        except Exception:
            self.meta_missing.configure(text="Missing values: (error)")

        # Duplicates for normalized view only – unknown at this point
        self.meta_dupes.configure(text="Duplicates (normalized): -")

    # ------------------------------------------------------------------ #
    # Normalization                                                      #
    # ------------------------------------------------------------------ #

    def _on_normalize(self) -> None:
        if not self.selected_file:
            messagebox.showinfo("No file", "Please select a CSV file first.")
            return

        try:
            # Use the class-based normalizer and force output into data/normalized
            normalized_root = ROOT / "data" / "normalized"
            from mikebot.data.normalize_csv import CSVNormalizer

            normalizer = CSVNormalizer(normalized_root)
            result = normalizer.normalize(self.selected_file)

            out_path = Path(result["output_path"])
            self.normalized_path = out_path
            self._log(f"Normalization succeeded → {out_path}")

            # Load normalized preview directly from the output path
            try:
                df = pd.read_csv(out_path)
                self._show_normalized_preview(df)
            except Exception as e:
                traceback_str = traceback.format_exc()
                messagebox.showerror("Error loading normalized CSV", str(e))
                self._log(f"Error loading normalized CSV: {e}\n{traceback_str}")

            # Auto-train if enabled
            if self.auto_train_var.get():
                self._on_train()

        except Exception as e:
            traceback_str = traceback.format_exc()
            messagebox.showerror("Normalization failed", str(e))
            self._log(f"Normalization failed: {e}\n{traceback_str}")

    def _on_refresh_normalized_preview(self) -> None:
        if not self.selected_symbol:
            messagebox.showinfo(
                "No symbol",
                "No symbol detected. Select a file first.",
            )
            return

        symbol_dir = DATA_NORMALIZED / self.selected_symbol

        if not symbol_dir.exists():
            messagebox.showinfo(
                "No normalized file",
                f"No normalized folder found:\n{symbol_dir}",
            )
            self._log(f"No normalized folder found: {symbol_dir}")
            return

        candidates = list(symbol_dir.glob("*.normalized.csv"))
        if not candidates:
            messagebox.showinfo(
                "No normalized file",
                f"No normalized CSV found in:\n{symbol_dir}",
            )
            self._log(f"No normalized CSV found in {symbol_dir}")
            return

        norm_path = max(candidates, key=lambda p: p.stat().st_mtime)
        self.normalized_path = norm_path

        try:
            df = pd.read_csv(norm_path)
            self._log(
                f"Loaded normalized data: {norm_path} "
                f"({len(df)} rows, {len(df.columns)} columns)"
            )
            self._show_normalized_preview(df)
        except Exception as e:
            messagebox.showerror("Error loading normalized file", str(e))
            self._log(f"Error loading normalized file: {e}")

    def _show_normalized_preview(self, df: pd.DataFrame) -> None:
        """
        Bulletproof normalized preview renderer.
        Ensures the normalized tree and metadata update consistently.
        """
        try:
            if df is None or df.empty:
                self._log("Normalized preview: empty or None DataFrame")
                messagebox.showinfo("No data", "Normalized file is empty or unreadable.")
                return

            self.norm_df = df
            self._update_norm_tree(df)
            self._update_metadata_from_normalized(df)
            self._log(
                f"Normalized preview updated: {len(df)} rows, {len(df.columns)} columns"
            )
        except Exception as e:
            self._log(f"Error showing normalized preview: {e}")
            messagebox.showerror("Preview error", str(e))

    def _update_norm_tree(self, df: pd.DataFrame, max_rows: int = 200) -> None:
        self._populate_tree(self.norm_tree, df, max_rows)

    def _update_metadata_from_normalized(self, df: pd.DataFrame) -> None:
        # Duplicates (simple heuristic)
        try:
            dupes = int(df.duplicated().sum())
            self.meta_dupes.configure(text=f"Duplicates (normalized): {dupes}")
        except Exception:
            self.meta_dupes.configure(text="Duplicates (normalized): (error)")

    # ------------------------------------------------------------------ #
    # Training                                                           #
    # ------------------------------------------------------------------ #

    def _on_train(self) -> None:
        if not VENV_PY.exists():
            messagebox.showerror(
                "Python executable not found",
                f"Expected venv python at:\n{VENV_PY}",
            )
            return

        if not self.selected_symbol:
            messagebox.showinfo(
                "No symbol",
                "Select a file first so the symbol can be detected.",
            )
            return

        if not self.normalized_path:
            messagebox.showinfo(
                "No normalized file",
                "Normalize a file or refresh the normalized preview first.",
            )
            return

        timeframe = (self.selected_timeframe or detect_timeframe_from_symbol(self.selected_symbol))

        cmd = [
            str(VENV_PY),
            "-m",
            "mikebot.core.train_pipeline",
            "--symbol",
            self.selected_symbol,
            "--timeframe",
            timeframe,
            "--normalized-file",
            str(self.normalized_path),
        ]

        try:
            self._log(f"Running training: {' '.join(cmd)}")

            out = subprocess.check_output(
                cmd,
                stderr=subprocess.STDOUT,
                timeout=1800,
            ).decode("utf-8", "ignore")

            if out.strip():
                self._log(f"Training output:\n{out.strip()}")
            else:
                self._log("Training completed with no stdout.")

        except subprocess.CalledProcessError as e:
            msg = e.output.decode("utf-8", "ignore") if e.output else str(e)
            messagebox.showerror("Training failed", msg)
            self._log(f"Training failed: {msg}")

        except Exception as e:
            traceback_str = traceback.format_exc()
            messagebox.showerror("Training error", str(e))
            self._log(f"Training error: {e}\n{traceback_str}")

    # ------------------------------------------------------------------ #
    # Tree population helper                                             #
    # ------------------------------------------------------------------ #

    def _populate_tree(
        self,
        tree: ttk.Treeview,
        df: pd.DataFrame,
        max_rows: int = 200,
    ) -> None:
        # Clear existing
        tree.delete(*tree.get_children())

        # Set columns
        cols = list(df.columns)
        tree["columns"] = cols
        tree["show"] = "headings"

        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=120, anchor="w")

        # Insert rows (limited)
        rows = df.head(max_rows).itertuples(index=False, name=None)
        for row in rows:
            values = [
                "" if (v is None or (isinstance(v, float) and pd.isna(v))) else v
                for v in row
            ]
            tree.insert("", "end", values=values)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def main() -> None:
    root = tk.Tk()
    root.title("Mikebot Studio – Data Ingestion & Training")
    root.geometry("1400x800")

    style = ttk.Style(root)
    try:
        style.theme_use("vista")
    except Exception:
        pass

    app = DataTab(root)
    app.pack(fill="both", expand=True)

    root.mainloop()


if __name__ == "__main__":
    main()