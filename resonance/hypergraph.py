from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set


@dataclass
class Node:
    """Node in a hypergraph."""

    id: str
    data: dict | None = None


class HyperGraph:
    """Minimal hypergraph with node and hyperedge management."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Set[str]] = []
        self._order: List[str] = []

    def add_node(
        self,
        node_id: str,
        data: Optional[dict] = None,
        connect: Optional[Iterable[str]] = None,
    ) -> None:
        """Add a node and optional hyperedge connecting ``connect``."""

        self.nodes[node_id] = Node(node_id, data or {})
        self._order.append(node_id)
        if connect:
            edge = set(connect)
            edge.add(node_id)
            self.edges.append(edge)

    def get_node(self, node_id: str) -> Node | None:
        """Return node with ``node_id`` if present."""

        return self.nodes.get(node_id)

    def neighbors(self, node_id: str) -> List[str]:
        """Return neighboring node ids connected via any hyperedge."""

        result: Set[str] = set()
        for edge in self.edges:
            if node_id in edge:
                result.update(edge)
        result.discard(node_id)
        return list(result)

    def trail(self, limit: int) -> List[str]:
        """Return ``limit`` most recently added node ids."""

        return self._order[-limit:]
