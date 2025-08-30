"""Differentiable in-memory graph store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from autograd import numpy as anp

from morphology import encode as encode_morph


@dataclass
class MemoryNode:
    speaker: str
    text: str
    index: int


class MemoryGraphStore:
    """Store dialogues with an embedding matrix for gradient-based retrieval."""

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim
        self.graph: Dict[str, List[MemoryNode]] = {}
        self.embeddings = anp.zeros((0, dim))

    def add_utterance(self, dialogue_id: str, speaker: str, text: str) -> anp.ndarray:
        """Encode ``text`` and append it to ``dialogue_id``.

        Returns the stored embedding which participates in gradients when
        ``retrieve`` is used with Autograd.
        """

        vec = anp.array(encode_morph(text, self.dim))
        if self.embeddings.size:
            self.embeddings = anp.vstack([self.embeddings, vec])
        else:
            self.embeddings = vec.reshape(1, -1)
        idx = self.embeddings.shape[0] - 1
        self.graph.setdefault(dialogue_id, []).append(MemoryNode(speaker, text, idx))
        return self.embeddings[idx]

    def get_dialogue(self, dialogue_id: str) -> List[MemoryNode]:
        return list(self.graph.get(dialogue_id, []))

    def embeddings_for(self, dialogue_id: str, speaker: Optional[str] = None) -> anp.ndarray:
        idxs = [
            n.index
            for n in self.graph.get(dialogue_id, [])
            if speaker is None or n.speaker == speaker
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
        # numerical stability
        scores = scores - anp.max(scores)
        weights = anp.exp(scores)
        weights = weights / anp.sum(weights)
        return weights @ vecs


class GraphRetriever:
    """Helper that provides high level queries over :class:`MemoryGraphStore`."""

    def __init__(self, store: MemoryGraphStore) -> None:
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
