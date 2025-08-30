"""Lightweight graph-based memory store.

This module defines two classes:

* :class:`MemoryGraphStore` - a tiny in-memory graph where dialogue turns
  are stored as nodes. Each dialogue is represented as a list of nodes but the
  structure can be easily extended with richer relations.
* :class:`GraphRetriever` - helper that looks up information from the
  ``MemoryGraphStore``.

The implementation avoids external dependencies and can optionally persist
its state to disk via JSON serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import os
from typing import Dict, List, Optional

from morphology import encode as encode_morph


@dataclass
class MemoryNode:
    """Single utterance within a dialogue."""

    speaker: str
    text: str
    morph_codes: List[float] = field(default_factory=list)


class MemoryGraphStore:
    """Store dialogues as simple graphs."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self.graph: Dict[str, List[MemoryNode]] = {}
        if path and os.path.exists(path):
            self.load(path)

    def add_utterance(self, dialogue_id: str, speaker: str, text: str) -> None:
        """Append an utterance to a dialogue and persist to disk."""

        codes = encode_morph(text).tolist()
        self.graph.setdefault(dialogue_id, []).append(
            MemoryNode(speaker, text, codes)
        )
        if self.path:
            self.save(self.path)

    def get_dialogue(self, dialogue_id: str) -> List[MemoryNode]:
        """Return the list of utterances for *dialogue_id*."""

        return list(self.graph.get(dialogue_id, []))

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


class GraphRetriever:
    """Small helper to fetch facts from :class:`MemoryGraphStore`."""

    def __init__(self, store: MemoryGraphStore) -> None:
        self.store = store

    def last_message(self, dialogue_id: str, speaker: str) -> Optional[str]:
        """Return the most recent message for ``speaker`` in ``dialogue_id``."""

        dialogue = self.store.get_dialogue(dialogue_id)
        for node in reversed(dialogue):
            if node.speaker == speaker:
                return node.text
        return None

    def all_messages(self, dialogue_id: str) -> List[str]:
        """Return all message texts from ``dialogue_id``."""

        return [node.text for node in self.store.get_dialogue(dialogue_id)]
