"""Context-based subgraph selection for routing."""

from __future__ import annotations

from typing import Iterable

from resonance.hypergraph import HyperGraph


def select_context_subgraph(
    graph: HyperGraph, context: Iterable[str]
) -> HyperGraph:
    """Return subgraph containing nodes matching ``context`` words."""

    keywords = {w.lower() for w in context}
    sub = HyperGraph()
    for node_id, node in graph.nodes.items():
        content = str(node.data.get("content", "")).lower()
        if any(k in content for k in keywords):
            sub.add_node(node_id, node.data)
            for edge in graph.edges:
                if node_id in edge:
                    sub.edges.append(set(edge))
    return sub
