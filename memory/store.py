"""Unified graph-based memory store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

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

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim
        self.graph: Dict[str, List[MemoryNode]] = {}
        self.embeddings = anp.zeros((0, dim))

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
        dialogue = self.graph.setdefault(dialogue_id, [])
        for node in dialogue:
            if node.speaker == speaker and node.text == text:
                return None

        if isinstance(embedding, dict):
            node = MemoryNode(speaker, text, embedding=embedding)
            dialogue.append(node)
            return None

        vec = anp.array(
            embedding if embedding is not None else encode_morph(text, self.dim)
        )
        if self.embeddings.size:
            self.embeddings = anp.vstack([self.embeddings, vec])
        else:
            self.embeddings = vec.reshape(1, -1)
        idx = self.embeddings.shape[0] - 1
        node = MemoryNode(speaker, text, index=idx, morph_codes=vec.tolist())
        dialogue.append(node)
        return self.embeddings[idx]

    def get_dialogue(self, dialogue_id: str) -> List[MemoryNode]:
        return list(self.graph.get(dialogue_id, []))

    # ---------------------------------------------------------------- embeddings
    def embeddings_for(
        self, dialogue_id: str, speaker: Optional[str] = None
    ) -> anp.ndarray:
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

        vecs = (
            embeddings
            if embeddings is not None
            else self.embeddings_for(dialogue_id, speaker)
        )
        if vecs.shape[0] == 0:
            return anp.zeros(self.dim)
        scores = vecs @ query
        scores = scores - anp.max(scores)
        weights = anp.exp(scores)
        weights = weights / anp.sum(weights)
        return weights @ vecs

    def most_similar(
        self,
        query: anp.ndarray,
        top_k: int,
        dialogue_id: str,
        speaker: Optional[str] = None,
    ) -> List[str]:
        """Return texts of the ``top_k`` most similar utterances to ``query``."""

        nodes = [
            n
            for n in self.graph.get(dialogue_id, [])
            if n.index is not None and (speaker is None or n.speaker == speaker)
        ]
        if not nodes:
            return []
        vecs = self.embeddings[[n.index for n in nodes]]
        norms = anp.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
        vecs_norm = vecs / norms
        q_norm = query / (anp.linalg.norm(query) + 1e-10)
        sims = vecs_norm @ q_norm
        top = anp.argsort(sims)[-top_k:][::-1]
        return [nodes[i].text for i in top]


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
