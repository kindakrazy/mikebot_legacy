# mikebot/ui/developer_tab.py
r"""
Developer Settings tab for Mikebot Studio.

This file is a standalone Tkinter application that will later be embedded
as a tab inside studio.py. It loads JSON configuration files from:

    - C:/mikebot/config          (modern)
    - C:/mikebot/config/legacy   (legacy, read-only by default)

and presents them in a dense, editable UI.

All paths in comments use forward slashes to avoid escape issues on Windows.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import tkinter as tk
from tkinter import ttk, messagebox

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG_FILE_NAMES = [
    "config.json",
    "control.json",
    "feeds.json",
    "global_config.json",
    "guardrails.json",
    "guardrails_schema.json",
    "health_monitor.json",
    "liveconfig.json",
    "mlorchestrator.json",
    "model_registry.json",
    "personality.json",
    "switches.json",
    "symbols.json",
]

API_STATE_URL = "http://127.0.0.1:8000/api/state"


def discover_config_dirs() -> List[Tuple[str, Path]]:
    """
    Return labeled config directories:
    - ("modern", C:/mikebot/config)
    - ("legacy", C:/mikebot/config/legacy)
    """
    here = Path(__file__).resolve()
    repo_root = here.parent.parent  # C:/mikebot

    modern = repo_root / "config"
    legacy = modern / "legacy"

    dirs: List[Tuple[str, Path]] = []
    if modern.exists():
        dirs.append(("modern", modern))
    if legacy.exists():
        dirs.append(("legacy", legacy))

    return dirs


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FlatEntry:
    path: str
    value: Any
    value_type: str


def flatten_config(data: Any, prefix: str = "") -> Dict[str, FlatEntry]:
    """
    Flatten nested dict/list structures into dotted paths.
    """
    flat: Dict[str, FlatEntry] = {}

    def _walk(node: Any, base: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                new_base = f"{base}.{k}" if base else k
                _walk(v, new_base)
        elif isinstance(node, list):
            for idx, v in enumerate(node):
                new_base = f"{base}[{idx}]"
                _walk(v, new_base)
        else:
            flat[base] = FlatEntry(
                path=base,
                value=node,
                value_type=type(node).__name__,
            )

    _walk(data, prefix)
    return flat


def set_nested_value(root: Any, path: str, new_value: Any) -> None:
    """
    Update nested dict/list given a flattened path like "a.b[0].c".
    """
    tokens: List[str] = []
    remaining = path

    while remaining:
        if "[" in remaining:
            before, rest = remaining.split("[", 1)
            if before:
                tokens.append(before)
            idx_str, after = rest.split("]", 1)
            tokens.append(f"[{idx_str}]")
            remaining = after[1:] if after.startswith(".") else after
        else:
            tokens.append(remaining)
            remaining = ""

    current = root
    for tok in tokens[:-1]:
        if tok.startswith("[") and tok.endswith("]"):
            idx = int(tok[1:-1])
            current = current[idx]
        else:
            current = current[tok]

    leaf = tokens[-1]
    if leaf.startswith("[") and leaf.endswith("]"):
        idx = int(leaf[1:-1])
        current[idx] = new_value
    else:
        current[leaf] = new_value


def parse_typed_value(raw: str, original_type: str) -> Any:
    """
    Convert user input into a Python value based on the original type.
    """
    s = raw.strip()

    if original_type == "bool":
        if s.lower() in ("true", "1", "yes", "on"):
            return True
        if s.lower() in ("false", "0", "no", "off"):
            return False

    if original_type == "int":
        try:
            return int(s)
        except ValueError:
            pass

    if original_type == "float":
        try:
            return float(s)
        except ValueError:
            pass

    try:
        return json.loads(s)
    except Exception:
        return raw


# ---------------------------------------------------------------------------
# Developer Settings Tab
# ---------------------------------------------------------------------------

class DeveloperSettingsTab(ttk.Frame):
    """
    High-density configuration editor for mikebot config files.

    - modern configs:  C:/mikebot/config
    - legacy configs:  C:/mikebot/config/legacy (read-only by default)

    Extended with a live "Minions / Survivability" subpanel that polls
    the Orchestrator API (/api/state) and renders:
      - minions: last decisions
      - survivability: safe mode, exposure/drawdown/volatility, quarantines
    """

    def __init__(self, master: tk.Misc, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)

        self.config_dirs: List[Tuple[str, Path]] = discover_config_dirs()

        self.current_file: Path | None = None
        self.current_section: str | None = None  # "modern" or "legacy"
        self.current_data: Any = None
        self.flat_entries: Dict[str, FlatEntry] = {}
        self.dirty: bool = False

        self.allow_legacy_editing: tk.BooleanVar = tk.BooleanVar(value=False)

        # Live state widgets (initialized in _build_ui)
        self.minions_text: tk.Text | None = None
        self.surv_text: tk.Text | None = None

        self._build_ui()
        self._load_config_list()

        # Start periodic polling of orchestrator state (if API is up)
        self._start_state_polling()

    # ----------------------------------------------------------------------
    # UI Construction
    # ----------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=3)
        self.columnconfigure(2, weight=2)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        # Left: config file list
        left = ttk.Frame(self)
        left.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(8, 4), pady=8)
        left.rowconfigure(1, weight=1)

        ttk.Label(left, text="Config files", font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, sticky="w"
        )

        self.config_listbox = tk.Listbox(
            left, exportselection=False, activestyle="dotbox"
        )
        self.config_listbox.grid(row=1, column=0, sticky="nsew", pady=(4, 4))
        self.config_listbox.bind("<<ListboxSelect>>", self._on_config_selected)

        # Legacy editing toggle
        self.legacy_toggle = ttk.Checkbutton(
            left,
            text="Allow editing legacy configs (dangerous)",
            variable=self.allow_legacy_editing,
            command=self._on_legacy_toggle,
        )
        self.legacy_toggle.grid(row=2, column=0, sticky="w", pady=(4, 0))

        # Center: table + edit panel
        center = ttk.Frame(self)
        center.grid(row=0, column=1, sticky="nsew", padx=4, pady=8)
        center.rowconfigure(1, weight=1)

        header = ttk.Frame(center)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        self.current_file_label = ttk.Label(
            header, text="No file loaded", font=("Segoe UI", 10, "bold")
        )
        self.current_file_label.grid(row=0, column=0, sticky="w")

        self.dirty_label = ttk.Label(
            header, text="", foreground="orange", font=("Segoe UI", 9, "italic")
        )
        self.dirty_label.grid(row=0, column=1, sticky="e", padx=(8, 0))

        self.section_label = ttk.Label(
            header, text="", foreground="gray", font=("Segoe UI", 9, "italic")
        )
        self.section_label.grid(row=0, column=2, sticky="e", padx=(8, 0))

        ttk.Button(header, text="Reload", command=self._reload_current).grid(
            row=0, column=3, padx=(8, 0)
        )
        ttk.Button(header, text="Save", command=self._save_current).grid(
            row=0, column=4, padx=(4, 0)
        )

        # Table
        table_frame = ttk.Frame(center)
        table_frame.grid(row=1, column=0, sticky="nsew", pady=(8, 4))
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        columns = ("path", "type", "value")
        self.tree = ttk.Treeview(
            table_frame, columns=columns, show="headings", selectmode="browse"
        )
        self.tree.heading("path", text="Setting")
        self.tree.heading("type", text="Type")
        self.tree.heading("value", text="Value")
        self.tree.column("path", width=260, anchor="w")
        self.tree.column("type", width=80, anchor="center")
        self.tree.column("value", width=200, anchor="w")
        self.tree.grid(row=0, column=0, sticky="nsew")

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=vsb.set)

        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # Edit panel
        edit = ttk.LabelFrame(center, text="Edit selected setting")
        edit.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        edit.columnconfigure(1, weight=1)

        ttk.Label(edit, text="Key:").grid(row=0, column=0, sticky="e", padx=4, pady=2)
        self.edit_key_var = tk.StringVar()
        ttk.Entry(edit, textvariable=self.edit_key_var, state="readonly").grid(
            row=0, column=1, sticky="ew", padx=4, pady=2
        )

        ttk.Label(edit, text="Current value:").grid(
            row=1, column=0, sticky="e", padx=4, pady=2
        )
        self.edit_current_var = tk.StringVar()
        ttk.Entry(edit, textvariable=self.edit_current_var, state="readonly").grid(
            row=1, column=1, sticky="ew", padx=4, pady=2
        )

        ttk.Label(edit, text="New value:").grid(
            row=2, column=0, sticky="e", padx=4, pady=2
        )
        self.edit_new_var = tk.StringVar()
        self.edit_new_entry = ttk.Entry(edit, textvariable=self.edit_new_var)
        self.edit_new_entry.grid(row=2, column=1, sticky="ew", padx=4, pady=2)
        self.edit_new_entry.bind("<Return>", lambda e: self._apply_edit())

        self.apply_button = ttk.Button(edit, text="Apply", command=self._apply_edit)
        self.apply_button.grid(row=2, column=2, sticky="w", padx=4)

        # Right: detail + diff + live state panels
        right = ttk.Frame(self)
        right.grid(row=0, column=2, rowspan=2, sticky="nsew", padx=(4, 8), pady=8)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        ttk.Label(right, text="Setting details", font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, sticky="w"
        )

        self.detail_text = tk.Text(
            right, wrap="word", height=20, width=50, state="disabled", font=("Consolas", 9)
        )
        self.detail_text.grid(row=1, column=0, sticky="nsew", pady=(4, 4))

        detail_scroll = ttk.Scrollbar(right, orient="vertical", command=self.detail_text.yview)
        detail_scroll.grid(row=1, column=1, sticky="ns")
        self.detail_text.configure(yscrollcommand=detail_scroll.set)

        diff_frame = ttk.Frame(right)
        diff_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        diff_frame.columnconfigure(0, weight=1)

        self.diff_button = ttk.Button(
            diff_frame, text="Show modern vs legacy diff", command=self._show_diff
        )
        self.diff_button.grid(row=0, column=0, sticky="ew")

        # Live Minions / Survivability subpanel
        live_label = ttk.Label(
            right, text="Live Orchestrator – Minions / Survivability", font=("Segoe UI", 10, "bold")
        )
        live_label.grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))

        # Minions panel
        minions_frame = ttk.LabelFrame(right, text="Minions")
        minions_frame.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=(4, 0))
        minions_frame.rowconfigure(0, weight=1)
        minions_frame.columnconfigure(0, weight=1)

        self.minions_text = tk.Text(
            minions_frame,
            wrap="word",
            height=10,
            width=50,
            state="disabled",
            font=("Consolas", 9),
        )
        self.minions_text.grid(row=0, column=0, sticky="nsew", padx=4, pady=(2, 4))

        minions_scroll = ttk.Scrollbar(minions_frame, orient="vertical", command=self.minions_text.yview)
        minions_scroll.grid(row=0, column=1, sticky="ns")
        self.minions_text.configure(yscrollcommand=minions_scroll.set)

        # Survivability panel
        surv_frame = ttk.LabelFrame(right, text="Survivability")
        surv_frame.grid(row=5, column=0, columnspan=2, sticky="nsew", pady=(4, 0))
        surv_frame.rowconfigure(0, weight=1)
        surv_frame.columnconfigure(0, weight=1)

        self.surv_text = tk.Text(
            surv_frame,
            wrap="word",
            height=8,
            width=50,
            state="disabled",
            font=("Consolas", 9),
        )
        self.surv_text.grid(row=0, column=0, sticky="nsew", padx=4, pady=(2, 4))

        surv_scroll = ttk.Scrollbar(surv_frame, orient="vertical", command=self.surv_text.yview)
        surv_scroll.grid(row=0, column=1, sticky="ns")
        self.surv_text.configure(yscrollcommand=surv_scroll.set)

        # Bottom: log panel
        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.grid(row=1, column=1, sticky="ew", padx=4, pady=(0, 8))
        log_frame.columnconfigure(0, weight=1)

        self.log_text = tk.Text(
            log_frame, height=4, wrap="word", state="disabled", font=("Consolas", 8)
        )
        self.log_text.grid(row=0, column=0, sticky="ew")

        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scroll.set)

        # Initial state: editing controls disabled until a file is loaded
        self._update_edit_controls_enabled(False)

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_dirty(self, dirty: bool) -> None:
        self.dirty = dirty
        self.dirty_label.configure(text="(unsaved changes)" if dirty else "")

    def _short(self, value: Any, max_len: int = 80) -> str:
        s = repr(value)
        return s if len(s) <= max_len else s[: max_len - 3] + "..."

    def _update_edit_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.edit_new_entry.configure(state=state)
        self.apply_button.configure(state=state)

    def _is_current_legacy(self) -> bool:
        return self.current_section == "legacy"

    # ----------------------------------------------------------------------
    # Config list handling
    # ----------------------------------------------------------------------

    def _load_config_list(self) -> None:
        self.config_listbox.delete(0, "end")

        if not self.config_dirs:
            self._log("No config directories found.")
            return

        for section, cfg_dir in self.config_dirs:
            header = f"[{section}]"
            self.config_listbox.insert("end", header)

            for name in CONFIG_FILE_NAMES:
                path = cfg_dir / name
                label = f"{section}/{name}"
                if not path.exists():
                    label += "  [missing]"
                self.config_listbox.insert("end", label)

    def _get_selected_config_path(self) -> Tuple[str, Path] | None:
        sel = self.config_listbox.curselection()
        if not sel:
            return None

        label = self.config_listbox.get(sel[0])

        # Section header
        if label.startswith("[") and label.endswith("]"):
            return None

        # label: "modern/config.json" or "legacy/config.json"
        try:
            section, filename = label.split("/", 1)
        except ValueError:
            return None

        for sec_label, cfg_dir in self.config_dirs:
            if sec_label == section:
                return section, cfg_dir / filename

        return None

    def _on_config_selected(self, event: tk.Event) -> None:
        result = self._get_selected_config_path()
        if result is None:
            return

        section, path = result

        if not path.exists():
            messagebox.showerror("Missing file", f"File not found: {path}")
            return

        if self.dirty and not self._confirm_discard_changes():
            return

        self._load_config_file(section, path)

    # ----------------------------------------------------------------------
    # Loading and saving files
    # ----------------------------------------------------------------------

    def _load_config_file(self, section: str, path: Path) -> None:
        try:
            text = path.read_text(encoding="utf-8")
            data = json.loads(text)
        except Exception as e:
            messagebox.showerror("Error loading config", f"{path}\n\n{e}")
            self._log(f"Error loading {path}: {e}")
            return

        self.current_file = path
        self.current_section = section
        self.current_data = data
        self.flat_entries = flatten_config(data)
        self._populate_tree()
        self._set_dirty(False)

        self.current_file_label.configure(text=str(path))
        if section == "modern":
            self.section_label.configure(text="modern (active)")
        else:
            self.section_label.configure(text="legacy (read-only by default)")

        # Enable / disable edit controls based on section + toggle
        if section == "legacy" and not self.allow_legacy_editing.get():
            self._update_edit_controls_enabled(False)
        else:
            self._update_edit_controls_enabled(True)

        self._log(f"Loaded [{section}] {path}")

    def _populate_tree(self) -> None:
        self.tree.delete(*self.tree.get_children())
        for key in sorted(self.flat_entries.keys()):
            entry = self.flat_entries[key]
            display_value = self._short(entry.value)
            self.tree.insert("", "end", iid=key, values=(entry.path, entry.value_type, display_value))

    def _confirm_discard_changes(self) -> bool:
        return messagebox.askyesno(
            "Discard changes?",
            "You have unsaved changes. Discard them?",
            icon=messagebox.WARNING,
        )

    def _reload_current(self) -> None:
        if not self.current_file or not self.current_section:
            return
        if self.dirty and not self._confirm_discard_changes():
            return
        self._load_config_file(self.current_section, self.current_file)

    def _save_current(self) -> None:
        if not self.current_file or self.current_data is None:
            return

        if self._is_current_legacy() and not self.allow_legacy_editing.get():
            messagebox.showinfo(
                "Legacy config",
                "Legacy configs are read-only unless you enable the toggle.\n"
                "No changes were saved.",
            )
            return

        try:
            text = json.dumps(self.current_data, indent=2, sort_keys=True)
            self.current_file.write_text(text, encoding="utf-8")
            self._set_dirty(False)
            self._log(f"Saved [{self.current_section}] {self.current_file}")
        except Exception as e:
            traceback_str = traceback.format_exc()
            messagebox.showerror("Error saving file", f"{e}")
            self._log(f"Error saving {self.current_file}: {e}\n{traceback_str}")

    # ----------------------------------------------------------------------
    # Editing
    # ----------------------------------------------------------------------

    def _on_tree_select(self, event: tk.Event) -> None:
        selection = self.tree.selection()
        if not selection:
            return

        key = selection[0]
        entry = self.flat_entries.get(key)
        if not entry:
            return

        self.edit_key_var.set(entry.path)
        self.edit_current_var.set(self._short(entry.value, max_len=200))
        self.edit_new_var.set(self._suggest(entry.value))

        self._update_detail(entry)

    def _suggest(self, value: Any) -> str:
        if isinstance(value, (int, float, bool, str)) or value is None:
            return str(value)
        try:
            return json.dumps(value)
        except Exception:
            return repr(value)

    def _update_detail(self, entry: FlatEntry) -> None:
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")

        self.detail_text.insert("end", f"Setting: {entry.path}\n")
        self.detail_text.insert("end", f"Type:    {entry.value_type}\n\n")

        self.detail_text.insert("end", "Current value:\n")
        try:
            pretty = json.dumps(entry.value, indent=2, ensure_ascii=False)
        except TypeError:
            pretty = repr(entry.value)
        self.detail_text.insert("end", pretty + "\n\n")

        self.detail_text.insert("end", "Expected usage / options:\n")

        if entry.value_type == "bool":
            self.detail_text.insert("end", "- Boolean flag (true/false).\n")
        elif entry.value_type in ("int", "float"):
            self.detail_text.insert("end", "- Numeric value.\n")
        elif entry.value_type == "str":
            self.detail_text.insert("end", "- Text value.\n")
        elif entry.value_type in ("list", "dict"):
            self.detail_text.insert("end", "- JSON structure ([] or {}).\n")
        else:
            self.detail_text.insert("end", "- Complex value.\n")

        key_lower = entry.path.lower()
        if "symbol" in key_lower:
            self.detail_text.insert("end", "- Symbol/instrument setting.\n")
        if "risk" in key_lower or "lot" in key_lower:
            self.detail_text.insert("end", "- Risk/positioning parameter.\n")
        if "path" in key_lower or "file" in key_lower:
            self.detail_text.insert("end", "- File/path parameter.\n")
        if "timeout" in key_lower or "interval" in key_lower:
            self.detail_text.insert("end", "- Time-related parameter.\n")
        if "guardrail" in key_lower:
            self.detail_text.insert("end", "- Guardrail/safety parameter.\n")

        if self._is_current_legacy():
            self.detail_text.insert(
                "end",
                "\nThis value comes from a LEGACY config.\n"
                "It reflects historical behavior, not the active defaults.\n",
            )

        self.detail_text.configure(state="disabled")

    def _apply_edit(self) -> None:
        if self.current_data is None or not self.current_section:
            return

        if self._is_current_legacy() and not self.allow_legacy_editing.get():
            messagebox.showinfo(
                "Legacy config",
                "Legacy configs are read-only unless you enable the toggle.",
            )
            return

        key = self.edit_key_var.get()
        if not key:
            return

        entry = self.flat_entries.get(key)
        if not entry:
            return

        new_raw = self.edit_new_var.get()
        if new_raw == "":
            if not messagebox.askyesno(
                "Empty value",
                "You are setting an empty value. Continue?",
            ):
                return

        new_value = parse_typed_value(new_raw, entry.value_type)

        try:
            set_nested_value(self.current_data, key, new_value)
        except Exception as e:
            traceback_str = traceback.format_exc()
            messagebox.showerror("Error applying change", f"{e}")
            self._log(f"Error applying change to {key}: {e}\n{traceback_str}")
            return

        entry.value = new_value
        entry.value_type = type(new_value).__name__
        self.flat_entries[key] = entry

        display_value = self._short(entry.value)
        self.tree.item(key, values=(entry.path, entry.value_type, display_value))

        self.edit_current_var.set(self._short(entry.value, max_len=200))
        self._update_detail(entry)
        self._set_dirty(True)

        self._log(f"Updated [{self.current_section}] {key} -> {repr(new_value)}")

    # ----------------------------------------------------------------------
    # Legacy toggle + diff
    # ----------------------------------------------------------------------

    def _on_legacy_toggle(self) -> None:
        if self._is_current_legacy():
            self._update_edit_controls_enabled(self.allow_legacy_editing.get())

    def _show_diff(self) -> None:
        """
        Show a modern vs legacy diff for the current file, if both exist.
        """
        if not self.current_file or not self.current_section:
            messagebox.showinfo("No file", "No config file is currently loaded.")
            return

        filename = self.current_file.name

        modern_dir = None
        legacy_dir = None
        for section, cfg_dir in self.config_dirs:
            if section == "modern":
                modern_dir = cfg_dir
            elif section == "legacy":
                legacy_dir = cfg_dir

        if modern_dir is None or legacy_dir is None:
            messagebox.showinfo(
                "Diff not available",
                "Both modern and legacy config directories are required for diff.",
            )
            return

        modern_path = modern_dir / filename
        legacy_path = legacy_dir / filename

        if not modern_path.exists() and not legacy_path.exists():
            messagebox.showinfo(
                "Diff not available",
                "Neither modern nor legacy version of this file exists.",
            )
            return

        try:
            modern_data = None
            legacy_data = None

            if modern_path.exists():
                modern_text = modern_path.read_text(encoding="utf-8")
                modern_data = json.loads(modern_text)

            if legacy_path.exists():
                legacy_text = legacy_path.read_text(encoding="utf-8")
                legacy_data = json.loads(legacy_text)
        except Exception as e:
            messagebox.showerror("Error loading for diff", str(e))
            return

        # Prepare pretty text
        modern_pretty = (
            json.dumps(modern_data, indent=2, ensure_ascii=False)
            if modern_data is not None
            else "<no modern version>"
        )
        legacy_pretty = (
            json.dumps(legacy_data, indent=2, ensure_ascii=False)
            if legacy_data is not None
            else "<no legacy version>"
        )

        # Show in a popup window
        win = tk.Toplevel(self)
        win.title(f"Diff: {filename} (modern vs legacy)")
        win.geometry("1200x700")

        win.columnconfigure(0, weight=1)
        win.columnconfigure(1, weight=1)
        win.rowconfigure(1, weight=1)

        ttk.Label(win, text="Modern", font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, sticky="w", padx=8, pady=(8, 0)
        )
        ttk.Label(win, text="Legacy", font=("Segoe UI", 10, "bold")).grid(
            row=0, column=1, sticky="w", padx=8, pady=(8, 0)
        )

        modern_text_widget = tk.Text(
            win, wrap="none", font=("Consolas", 9)
        )
        legacy_text_widget = tk.Text(
            win, wrap="none", font=("Consolas", 9)
        )

        modern_text_widget.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=(4, 8))
        legacy_text_widget.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=(4, 8))

        # Scrollbars
        m_vsb = ttk.Scrollbar(win, orient="vertical", command=modern_text_widget.yview)
        m_vsb.grid(row=1, column=0, sticky="nse", padx=(0, 0))
        modern_text_widget.configure(yscrollcommand=m_vsb.set)

        l_vsb = ttk.Scrollbar(win, orient="vertical", command=legacy_text_widget.yview)
        l_vsb.grid(row=1, column=1, sticky="nse", padx=(0, 0))
        legacy_text_widget.configure(yscrollcommand=l_vsb.set)

        m_hsb = ttk.Scrollbar(win, orient="horizontal", command=modern_text_widget.xview)
        m_hsb.grid(row=2, column=0, sticky="ew", padx=(8, 4))
        modern_text_widget.configure(xscrollcommand=m_hsb.set)

        l_hsb = ttk.Scrollbar(win, orient="horizontal", command=legacy_text_widget.xview)
        l_hsb.grid(row=2, column=1, sticky="ew", padx=(4, 8))
        legacy_text_widget.configure(xscrollcommand=l_hsb.set)

        modern_text_widget.insert("1.0", modern_pretty)
        legacy_text_widget.insert("1.0", legacy_pretty)

        modern_text_widget.configure(state="disabled")
        legacy_text_widget.configure(state="disabled")

        self._log(f"Opened diff for {filename}")
    # ---------------------------------------------------------------------------
    # Live orchestrator + strategy engine state polling (expanded)
    # ---------------------------------------------------------------------------

    def _start_state_polling(self) -> None:
        """
        Kick off periodic polling of the new Group B API endpoints.
        Uses Tkinter's after() to stay on the main thread.
        """
        self.after(1000, self._refresh_state_once)


    def _refresh_state_once(self) -> None:
        """
        Fetch /strategy/state and /system/state and update the Minions /
        Survivability panels. Reschedules itself via after().
        """
        from mikebot.ui.api_client import (
            get_strategy_state,
            get_system_state,
        )

        # Strategy engine + minions
        strat_data = get_strategy_state()
        # System-level survivability + orchestrator
        sys_data = get_system_state()

        # Update panels
        self._update_minions_panel(strat_data)
        self._update_survivability_panel(sys_data)

        # Poll again
        self.after(1000, self._refresh_state_once)


    def _update_minions_panel(self, strat: Any) -> None:
        """
        Render strategy engine + minion state in the Minions panel.
        strat is the dict returned by get_strategy_state().
        """
        if self.minions_text is None:
            return

        self.minions_text.configure(state="normal")
        self.minions_text.delete("1.0", "end")

        if not isinstance(strat, dict) or "error" in strat:
            self.minions_text.insert("1.0", f"<error>\n{strat.get('error') if isinstance(strat, dict) else strat}")
            self.minions_text.configure(state="disabled")
            return

        # Extract fields from /strategy/state
        payload = {
            "active_minions": strat.get("active_minions"),
            "last_minion_decisions": strat.get("last_minion_decisions"),
            "strategy_engine_state": strat.get("strategy_engine_state"),
            "errors": strat.get("errors"),
            "extras": strat.get("extras"),
            "context": strat.get("context"),
        }

        try:
            pretty = json.dumps(payload, indent=2, ensure_ascii=False)
        except Exception:
            pretty = repr(payload)

        self.minions_text.insert("1.0", pretty)
        self.minions_text.configure(state="disabled")


    def _update_survivability_panel(self, sys_state: Any) -> None:
        """
        Render survivability + orchestrator state in the Survivability panel.
        sys_state is the dict returned by get_system_state().
        """
        if self.surv_text is None:
            return

        self.surv_text.configure(state="normal")
        self.surv_text.delete("1.0", "end")

        if not isinstance(sys_state, dict) or "error" in sys_state:
            self.surv_text.insert("1.0", f"<error>\n{sys_state.get('error') if isinstance(sys_state, dict) else sys_state}")
            self.surv_text.configure(state="disabled")
            return

        # Extract fields from /system/state
        payload = {
            "regime": sys_state.get("regime"),
            "personality": sys_state.get("personality"),
            "orchestrator_status": sys_state.get("orchestrator_status"),
            "active_symbol": sys_state.get("active_symbol"),
            "active_model": sys_state.get("active_model"),
            "heartbeat": sys_state.get("heartbeat"),
            "last_tick": sys_state.get("last_tick"),
            "last_error": sys_state.get("last_error"),
            "survivability_guard": sys_state.get("survivability_guard"),
        }

        try:
            pretty = json.dumps(payload, indent=2, ensure_ascii=False)
        except Exception:
            pretty = repr(payload)

        self.surv_text.insert("1.0", pretty)
        self.surv_text.configure(state="disabled")

# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def main() -> None:
    root = tk.Tk()
    root.title("Mikebot Studio – Developer Settings")
    root.geometry("1200x700")

    style = ttk.Style(root)
    try:
        style.theme_use("vista")
    except Exception:
        pass

    app = DeveloperSettingsTab(root)
    app.pack(fill="both", expand=True)

    root.mainloop()


if __name__ == "__main__":
    main()
