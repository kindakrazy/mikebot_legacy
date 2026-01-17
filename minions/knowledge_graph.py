# C:\mikebot\minions\knowledge_graph.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class FeatureNode:
    name: str
    importance: float
    group: str


@dataclass
class StrategyNode:
    name: str
    features: List[str]


class KnowledgeGraph:
    """
    Lightweight in-memory knowledge graph.

    Distillation of HighStrike's knowledge_graph + graph_hook:
      - feature groups
      - strategy → feature mapping
      - importance weights
    """

    def __init__(
        self,
        features: Dict[str, FeatureNode],
        strategies: Dict[str, StrategyNode],
    ) -> None:
        self.features = features
        self.strategies = strategies

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "KnowledgeGraph":
        feat_nodes = {
            name: FeatureNode(
                name=name,
                importance=float(meta.get("importance", 1.0)),
                group=str(meta.get("group", "generic")),
            )
            for name, meta in cfg.get("features", {}).items()
        }
        strat_nodes = {
            name: StrategyNode(
                name=name,
                features=list(meta.get("features", [])),
            )
            for name, meta in cfg.get("strategies", {}).items()
        }
        return cls(feat_nodes, strat_nodes)

    def feature_weights_for_strategy(self, strategy_name: str) -> Dict[str, float]:
        strat = self.strategies.get(strategy_name)
        if not strat:
            return {}
        return {
            f: self.features.get(f, FeatureNode(f, 1.0, "generic")).importance
            for f in strat.features
        }

    def global_feature_weights(self) -> Dict[str, float]:
        return {name: node.importance for name, node in self.features.items()}
