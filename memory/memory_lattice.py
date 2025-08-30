"""Graph-based memory store with embeddings.

This is an extension of :mod:`memory_graph` where each node stores an
additional embedding vector.  The structure remains a simple mapping from
``dialogue_id`` to a list of nodes, but embeddings allow vector based
retrieval in combination with structural relations.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import os
from typing import Dict, List, Optional


@dataclass
class MemoryNode:
    """Single utterance with an embedding."""

    speaker: str
    text: str
    embedding: Dict[str, float]


class MemoryGraphStore:
    """Store dialogues as simple graphs with vector representations."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self.graph: Dict[str, List[MemoryNode]] = {}
        if path and os.path.exists(path):
            self.load(path)

    def add_utterance(
        self,
        dialogue_id: str,
        speaker: str,
        text: str,
        embedding: Dict[str, float],
    ) -> None:
        """Append an utterance with ``embedding`` to a dialogue."""

        self.graph.setdefault(dialogue_id, []).append(
            MemoryNode(speaker, text, embedding)
        )
        if self.path:
            self.save(self.path)

    def get_dialogue(self, dialogue_id: str) -> List[MemoryNode]:
        """Return the list of nodes for ``dialogue_id``."""

        return list(self.graph.get(dialogue_id, []))

    def all_nodes(self) -> List[MemoryNode]:
        """Return a flat list of all nodes across dialogues."""

        return [node for nodes in self.graph.values() for node in nodes]

    # Persistence -------------------------------------------------------
    def save(self, path: Optional[str] = None) -> None:
        path = path or self.path
        if not path:
            return
        data = {
            did: [asdict(node) for node in nodes]
            for did, nodes in self.graph.items()
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        self.path = path

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        self.graph = {
            did: [MemoryNode(**node) for node in nodes]
            for did, nodes in raw.items()
        }
        self.path = path
