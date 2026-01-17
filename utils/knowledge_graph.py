from __future__ import annotations

from typing import Dict, List, Iterable


class KnowledgeGraph:
    """
    Lightweight utility knowledge graph.

    This is intentionally minimal and decoupled from the minions KnowledgeGraph;
    it is used for ad‑hoc tooling and diagnostics.
    """

    def __init__(self) -> None:
        self._graph: Dict[str, List[str]] = {}

    def add_entity(self, entity: str) -> None:
        self._graph.setdefault(entity, [])

    def link(self, src: str, dst: str) -> None:
        self._graph.setdefault(src, []).append(dst)
        self._graph.setdefault(dst, [])

    def neighbors(self, entity: str) -> List[str]:
        return list(self._graph.get(entity, []))

    def entities(self) -> Iterable[str]:
        return self._graph.keys()

    def to_dict(self) -> Dict[str, List[str]]:
        return {k: list(v) for k, v in self._graph.items()}