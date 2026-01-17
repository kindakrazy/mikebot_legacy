# mikebot/ui/lineage_tab.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional

import tkinter as tk
from tkinter import ttk, messagebox

from mikebot.core.model_lineage import ModelLineageRegistry
from mikebot.core.model_lineage_visualizer import ModelLineageVisualizer
from mikebot.core.model_registry import ModelRegistry


class LineageTab(ttk.Frame):
    """
    Mikebot Studio - Lineage & Evolution Tab (v4, symbol-level MULTITF).

    Visual interface to explore the 'Family Tree' of trained models and compare
    performance across all symbols, scopes, and model types.

    v4 assumptions:
      - One primary scope: "MULTITF"
      - ModelRegistry is symbol-level and scope-aware
      - ModelLineageRegistry stores versions keyed by (symbol, timeframe, model_type)
    """

    def __init__(self, parent: tk.Misc, repo_root: Path):
        super().__init__(parent)
        self.repo_root = repo_root

        # Core registries
        self.registry_path = repo_root / "config" / "model_registry.json"
        self.lineage_path = repo_root / "config" / "model_lineage.json"

        self.model_reg = ModelRegistry(self.registry_path)
        self.lineage_reg = ModelLineageRegistry(self.lineage_path)
        self.visualizer = ModelLineageVisualizer(self.lineage_reg)

        # UI state variables
        self.symbol_var = tk.StringVar(value="")
        self.scope_var = tk.StringVar(value="MULTITF")
        self.type_var = tk.StringVar(value="")

        # Internal index: version_id -> node dict
        self._node_index: Dict[str, Dict[str, Any]] = {}

        self._build_ui()
        self._init_selectors_from_registry()

    # ------------------------------------------------------------------ #
    # UI construction                                                    #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        # Top control bar
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(fill="x", padx=10, pady=10)

        # Symbol selector
        ttk.Label(ctrl_frame, text="Symbol:").pack(side="left")
        self.symbol_combo = ttk.Combobox(
            ctrl_frame,
            textvariable=self.symbol_var,
            values=[],
            width=16,
            state="readonly",
        )
        self.symbol_combo.pack(side="left", padx=5)

        # Scope selector (v4: usually just MULTITF, but keep flexible)
        ttk.Label(ctrl_frame, text="Scope:").pack(side="left", padx=(10, 0))
        self.scope_combo = ttk.Combobox(
            ctrl_frame,
            textvariable=self.scope_var,
            values=[],
            width=10,
            state="readonly",
        )
        self.scope_combo.pack(side="left", padx=5)

        # Type selector
        ttk.Label(ctrl_frame, text="Type:").pack(side="left", padx=(10, 0))
        self.type_combo = ttk.Combobox(
            ctrl_frame,
            textvariable=self.type_var,
            values=[],
            width=10,
            state="readonly",
        )
        self.type_combo.pack(side="left", padx=5)

        # Refresh button
        ttk.Button(
            ctrl_frame,
            text="Refresh Tree",
            command=self.refresh,
        ).pack(side="left", padx=20)

        # React to selector changes
        self.symbol_combo.bind("<<ComboboxSelected>>", self._on_symbol_changed)
        self.scope_combo.bind("<<ComboboxSelected>>", self._on_scope_changed)
        self.type_combo.bind("<<ComboboxSelected>>", self._on_type_changed)

        # Main content (paned window)
        pw = ttk.PanedWindow(self, orient="horizontal")
        pw.pack(fill="both", expand=True, padx=10, pady=5)

        # Left: model family tree
        tree_frame = ttk.LabelFrame(pw, text="Model Family Tree")
        self.tree = ttk.Treeview(
            tree_frame,
            columns=("Type", "WR", "ACC"),
            show="tree headings",
        )
        self.tree.heading("#0", text="Version ID")
        self.tree.heading("Type", text="Experiment")
        self.tree.heading("WR", text="Win Rate")
        self.tree.heading("ACC", text="Accuracy")

        self.tree.column("#0", width=260, anchor="w")
        self.tree.column("Type", width=160, anchor="w")
        self.tree.column("WR", width=90, anchor="e")
        self.tree.column("ACC", width=90, anchor="e")
        self.tree.pack(fill="both", expand=True)
        pw.add(tree_frame, weight=3)

        # Right: evolution details
        details_frame = ttk.LabelFrame(pw, text="Evolution Details")
        self.details_text = tk.Text(
            details_frame,
            height=18,
            state="disabled",
            bg="#f0f0f0",
            font=("Consolas", 9),
        )
        self.details_text.pack(fill="both", expand=True, padx=5, pady=5)

        self.promote_btn = ttk.Button(
            details_frame,
            text="Manually Promote to Live",
            command=self._promote_selected,
        )
        self.promote_btn.pack(pady=10)

        pw.add(details_frame, weight=2)

        self.tree.bind("<<TreeviewSelect>>", self._on_select)

    # ------------------------------------------------------------------ #
    # Selector initialization                                           #
    # ------------------------------------------------------------------ #

    def _init_selectors_from_registry(self) -> None:
        """
        Initialize symbol, scope, and type selectors from the registry.
        Uses ModelRegistry.list_symbols() as the source of truth.
        """
        symbols = self.model_reg.get_symbol_list()
        self.symbol_combo["values"] = symbols

        if not symbols:
            self.symbol_var.set("")
            self.scope_var.set("")
            self.type_var.set("")
            return

        # Pick first symbol by default
        first_symbol = symbols[0]
        self.symbol_var.set(first_symbol)

        scopes = self.model_reg.get_scopes_for_symbol(first_symbol)
        self.scope_combo["values"] = scopes
        if scopes:
            # Prefer MULTITF if present
            scope = "MULTITF" if "MULTITF" in scopes else scopes[0]
            self.scope_var.set(scope)

            types = self.model_reg.get_types_for_symbol_scope(first_symbol, scope)
            self.type_combo["values"] = types
            if types:
                self.type_var.set(types[0])

        self.refresh()

    def _on_symbol_changed(self, event: Any) -> None:
        symbol = self.symbol_var.get()
        scopes = self.model_reg.get_scopes_for_symbol(symbol)
        self.scope_combo["values"] = scopes

        if scopes:
            if self.scope_var.get() not in scopes:
                scope = "MULTITF" if "MULTITF" in scopes else scopes[0]
                self.scope_var.set(scope)
            scope = self.scope_var.get()
            types = self.model_reg.get_types_for_symbol_scope(symbol, scope)
            self.type_combo["values"] = types
            if types and self.type_var.get() not in types:
                self.type_var.set(types[0])

        self.refresh()

    def _on_scope_changed(self, event: Any) -> None:
        symbol = self.symbol_var.get()
        scope = self.scope_var.get()
        types = self.model_reg.get_types_for_symbol_scope(symbol, scope)
        self.type_combo["values"] = types
        if types and self.type_var.get() not in types:
            self.type_var.set(types[0])
        self.refresh()

    def _on_type_changed(self, event: Any) -> None:
        self.refresh()

    # ------------------------------------------------------------------ #
    # Tree refresh                                                       #
    # ------------------------------------------------------------------ #

    def refresh(self) -> None:
        """Reloads lineage from disk and repopulates the tree."""
        # Clear tree
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._node_index.clear()

        symbol = self.symbol_var.get()
        scope = self.scope_var.get()
        m_type = self.type_var.get()

        if not symbol or not scope or not m_type:
            return

        # Reload underlying registries (in case training updated them)
        self.model_reg.reload()
        if hasattr(self.lineage_reg, "reload"):
            self.lineage_reg.reload()  # type: ignore[attr-defined]

        # v4: scope is the lineage timeframe (MULTITF)
        tree_data = self.visualizer.get_evolution_tree(symbol, scope, m_type)
        if not tree_data or "tree" not in tree_data or not tree_data["tree"]:
            return

        root_node = tree_data["tree"]

        # Allow synthetic root ("__ROOT__") – expand its children instead
        if root_node.get("id") == "__ROOT__":
            for child in root_node.get("children", []):
                self._insert_node("", child)
        else:
            self._insert_node("", root_node)

    def _insert_node(self, parent_id: str, node: Dict[str, Any]) -> None:
        """Recursively inserts nodes into the Tkinter Treeview."""
        version_id = node.get("id", "")
        metrics = node.get("metrics", {}) or {}

        wr = metrics.get("win_rate", None)
        acc = metrics.get("accuracy", None)

        wr_disp = f"{wr:.4f}" if isinstance(wr, (int, float)) else ""
        acc_disp = f"{acc:.4f}" if isinstance(acc, (int, float)) else ""

        item_id = self.tree.insert(
            parent_id,
            "end",
            text=version_id,
            values=(node.get("name", ""), wr_disp, acc_disp),
            open=True,
        )

        # Index node by version_id for detail view
        if version_id:
            self._node_index[version_id] = node

        for child in node.get("children", []):
            self._insert_node(item_id, child)

    # ------------------------------------------------------------------ #
    # Selection & details                                                #
    # ------------------------------------------------------------------ #

    def _on_select(self, event: Any) -> None:
        selected = self.tree.selection()
        if not selected:
            return

        item_id = selected[0]
        v_id = self.tree.item(item_id, "text")
        parent_id = self.tree.parent(item_id)

        symbol = self.symbol_var.get()
        scope = self.scope_var.get()
        m_type = self.type_var.get()

        lines: List[str] = []
        lines.append(f"Model Version: {v_id}")
        lines.append(f"Symbol:       {symbol}")
        lines.append(f"Scope:        {scope}")
        lines.append(f"Type:         {m_type}")

        values = self.tree.item(item_id, "values")
        exp_name = values[0] if values else ""
        wr_str = values[1] if len(values) > 1 else ""
        acc_str = values[2] if len(values) > 2 else ""

        lines.append("")
        lines.append(f"Experiment:   {exp_name}")
        if wr_str:
            lines.append(f"Win Rate:     {wr_str}")
        if acc_str:
            lines.append(f"Accuracy:     {acc_str}")

        # Evolution delta vs parent
        if parent_id:
            p_v_id = self.tree.item(parent_id, "text")
            if p_v_id and p_v_id != "__ROOT__":
                deltas = self.visualizer.get_performance_delta(
                    v_id, p_v_id, symbol, scope, m_type
                )
                if deltas:
                    lines.append("")
                    lines.append(f"--- Evolution from parent {p_v_id} ---")
                    for metric_name, delta in deltas.items():
                        sign = "+" if delta >= 0 else ""
                        lines.append(f"{metric_name}: {sign}{delta:.4f}")

        # Extended details from node_index
        node = self._node_index.get(v_id)
        if node:
            # Regime performance
            regime_perf = node.get("regime_performance", {}) or {}
            if regime_perf:
                lines.append("")
                lines.append("--- Regime Performance ---")
                for regime_label, metrics in regime_perf.items():
                    lines.append(f"[regime={regime_label}]")
                    for mk, mv in metrics.items():
                        lines.append(f"  {mk}: {mv}")
                    lines.append("")

            # Strategy performance
            strat_perf = node.get("strategy_performance", {}) or {}
            if strat_perf:
                lines.append("")
                lines.append("--- Strategy Performance ---")
                for strat_name, metrics in strat_perf.items():
                    lines.append(f"[strategy={strat_name}]")
                    for mk, mv in metrics.items():
                        lines.append(f"  {mk}: {mv}")
                    lines.append("")

            # Feature importance (top N)
            feat_imp = node.get("feature_importance", {}) or {}
            if feat_imp:
                lines.append("")
                lines.append("--- Feature Importance (top 10) ---")
                sorted_items = sorted(
                    feat_imp.items(), key=lambda kv: kv[1], reverse=True
                )[:10]
                for fname, val in sorted_items:
                    lines.append(f"{fname}: {val:.4f}")

            notes = node.get("notes")
            if notes:
                lines.append("")
                lines.append("--- Notes ---")
                lines.append(str(notes))

        info = "\n".join(lines)

        self.details_text.config(state="normal")
        self.details_text.delete("1.0", "end")
        self.details_text.insert("1.0", info)
        self.details_text.config(state="disabled")

    # ------------------------------------------------------------------ #
    # Promotion                                                          #
    # ------------------------------------------------------------------ #

    def _promote_selected(self) -> None:
        selected = self.tree.selection()
        if not selected:
            return

        v_id = self.tree.item(selected[0], "text")
        symbol = self.symbol_var.get()
        scope = self.scope_var.get()
        m_type = self.type_var.get()

        if not symbol or not scope or not m_type:
            messagebox.showinfo(
                "No selection",
                "Select symbol, scope, and type first.",
            )
            return

        if messagebox.askyesno(
            "Confirm Promotion",
            f"Promote {v_id} to active live model for {symbol} {scope} ({m_type})?",
        ):
            # v4: scope is the lineage timeframe; registry is symbol-level MULTITF
            # We treat scope as timeframe for lineage, but ModelRegistry only needs symbol + type.
            try:
                self.model_reg.update_active_version(symbol, m_type, v_id)
                messagebox.showinfo(
                    "Success",
                    f"Model {v_id} is now LIVE for {symbol} {scope} ({m_type}).",
                )
            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"Failed to promote model {v_id}: {e}",
                )


# --------------------------------------------------------------------------- #
# Standalone runner                                                           #
# --------------------------------------------------------------------------- #

def main() -> None:
    root = tk.Tk()
    root.title("Mikebot Studio – Model Lineage & Evolution (v4)")
    root.geometry("1400x900")

    style = ttk.Style(root)
    try:
        style.theme_use("vista")
    except Exception:
        pass

    HERE = Path(__file__).resolve()
    repo_root = HERE.parent.parent  # mikebot/

    app = LineageTab(root, repo_root)
    app.pack(fill="both", expand=True)

    root.mainloop()


if __name__ == "__main__":
    main()
