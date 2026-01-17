# mikebot/core/model_lineage_visualizer.py

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
from mikebot.core.model_lineage import ModelLineageRegistry

logger = logging.getLogger(__name__)

class ModelLineageVisualizer:
    """
    Translates flat lineage records into hierarchical structures 
    for UI visualization and performance auditing.
    """

    def __init__(self, registry: ModelLineageRegistry):
        self.registry = registry

    def get_evolution_tree(self, symbol: str, timeframe: str, model_type: str) -> Dict[str, Any]:
        """
        Builds a recursive dictionary representing the 'Family Tree' 
        of models for a specific asset.
        """
        all_lineage = self.registry.load_lineage()
        key = f"{symbol}_{timeframe}_{model_type}"
        
        if key not in all_lineage:
            return {"symbol": symbol, "tree": None}

        history = all_lineage[key]
        # Sort by creation to find the root (oldest)
        versions = sorted(history.items(), key=lambda x: x[1]['timestamp'])
        
        if not versions:
            return {"symbol": symbol, "tree": None}

        # Build map of Parent -> Children
        nodes = {}
        root_id = None
        
        for v_id, meta in history.items():
            nodes[v_id] = {
                "id": v_id,
                "name": meta.get("experiment_type", "base"),
                "metrics": meta.get("metrics", {}),
                "children": []
            }
            if not meta.get("parent_id"):
                root_id = v_id

        # Link children to parents
        for v_id, meta in history.items():
            parent_id = meta.get("parent_id")
            if parent_id in nodes:
                nodes[parent_id]["children"].append(nodes[v_id])

        return {
            "symbol": symbol,
            "root_version": root_id,
            "tree": nodes.get(root_id)
        }

    def get_performance_delta(
        self, child_id: str, parent_id: str, symbol: str, tf: str, m_type: str
    ) -> Dict[str, float]:
        """
        Calculates the improvement (or regression) between two generations.
        Used by the UI to highlight 'IQ Boosts'.
        """
        all_lineage = self.registry.load_lineage()
        key = f"{symbol}_{tf}_{m_type}"
        history = all_lineage.get(key, {})

        child = history.get(child_id, {})
        parent = history.get(parent_id, {})

        if not child or not parent:
            return {}

        c_metrics = child.get("metrics", {})
        p_metrics = parent.get("metrics", {})

        deltas = {}
        for metric in ["wr", "ev", "pf"]:
            c_val = c_metrics.get(metric, 0)
            p_val = p_metrics.get(metric, 0)
            deltas[metric] = c_val - p_val

        return deltas