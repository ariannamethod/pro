"""Unified graph-based memory store."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import json
import os

from autograd import numpy as anp

from morphology import encode as encode_morph


@dataclass
class MemoryNode:
    """Single utterance optionally backed by an embedding vector."""

    speaker: str
    text: str
    index: Optional[int] = None
    embedding: Optional[Dict[str, float]] = None
    morph_codes: Optional[List[float]] = None


class MemoryStore:
    """Store dialogues with optional embeddings for retrieval."""

    def __init__(self, dim: int = 32, path: Optional[str] = None) -> None:
        self.dim = dim
        self.path = path
        self.graph: Dict[str, List[MemoryNode]] = {}
        self.embeddings = anp.zeros((0, dim))
        if path and os.path.exists(path):
            self.load(path)

    # ------------------------------------------------------------------ storage
    def add_utterance(
        self,
        dialogue_id: str,
        speaker: str,
        text: str,
        embedding: Optional[Any] = None,
    ) -> Optional[anp.ndarray]:
        """Append an utterance to a dialogue.

        If ``embedding`` is a dictionary it is stored directly with the node.
        Otherwise the text is encoded (or the provided vector is used) and the
        resulting embedding participates in gradient-based retrieval.
        """

        if isinstance(embedding, dict):
            node = MemoryNode(speaker, text, embedding=embedding)
            self.graph.setdefault(dialogue_id, []).append(node)
            if self.path:
                self.save(self.path)
            return None

        vec = anp.array(embedding if embedding is not None else encode_morph(text, self.dim))
        if self.embeddings.size:
            self.embeddings = anp.vstack([self.embeddings, vec])
        else:
            self.embeddings = vec.reshape(1, -1)
        idx = self.embeddings.shape[0] - 1
        node = MemoryNode(speaker, text, index=idx, morph_codes=vec.tolist())
        self.graph.setdefault(dialogue_id, []).append(node)
        return self.embeddings[idx]

    def get_dialogue(self, dialogue_id: str) -> List[MemoryNode]:
        return list(self.graph.get(dialogue_id, []))

    # ---------------------------------------------------------------- embeddings
    def embeddings_for(self, dialogue_id: str, speaker: Optional[str] = None) -> anp.ndarray:
        idxs = [
            n.index
            for n in self.graph.get(dialogue_id, [])
            if n.index is not None and (speaker is None or n.speaker == speaker)
        ]
        if not idxs:
            return anp.zeros((0, self.dim))
        return self.embeddings[idxs]

    def retrieve(
        self,
        query: anp.ndarray,
        dialogue_id: str,
        speaker: Optional[str] = None,
        embeddings: Optional[anp.ndarray] = None,
    ) -> anp.ndarray:
        """Return a weighted memory vector for ``query`` using softmax weighting."""

        vecs = embeddings if embeddings is not None else self.embeddings_for(dialogue_id, speaker)
        if vecs.shape[0] == 0:
            return anp.zeros(self.dim)
        scores = vecs @ query
        scores = scores - anp.max(scores)
        weights = anp.exp(scores)
        weights = weights / anp.sum(weights)
        return weights @ vecs

    # ---------------------------------------------------------------- persistence
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
        self.graph = {}
        vecs: List[List[float]] = []
        for did, nodes in raw.items():
            restored: List[MemoryNode] = []
            for node in nodes:
                mn = MemoryNode(**node)
                if mn.morph_codes is not None:
                    mn.index = len(vecs)
                    vecs.append(mn.morph_codes)
                restored.append(mn)
            self.graph[did] = restored
        self.embeddings = anp.array(vecs) if vecs else anp.zeros((0, self.dim))
        self.path = path


class GraphRetriever:
    """Helper that provides high level queries over :class:`MemoryStore`."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def last_message(self, dialogue_id: str, speaker: str) -> Optional[str]:
        for node in reversed(self.store.get_dialogue(dialogue_id)):
            if node.speaker == speaker:
                return node.text
        return None

    def all_messages(self, dialogue_id: str) -> List[str]:
        return [n.text for n in self.store.get_dialogue(dialogue_id)]

    def retrieve(
        self, dialogue_id: str, speaker: str, query: Optional[anp.ndarray] = None
    ) -> anp.ndarray:
        if query is None:
            query = anp.zeros(self.store.dim)
        return self.store.retrieve(query, dialogue_id, speaker)
