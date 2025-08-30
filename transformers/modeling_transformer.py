"""Minimal transformer building blocks used in tests and demos.

This file introduces :class:`MemoryAttention`, a simple mechanism that
injects information retrieved from a memory graph into a sequence of hidden
states.  By default it works with :class:`~memory.memory_graph.GraphRetriever`
but it can also consume a :class:`~memory.reinforce_retriever.ReinforceRetriever`
whose probability distribution over nodes defines a soft cross-attention.
The goal is not to implement a full Transformer model but to provide a
lightweight hook where a memory graph can influence the computation.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from memory.memory_graph import GraphRetriever
from memory.reinforce_retriever import ReinforceRetriever


class MemoryAttention:
    """Additively combines hidden states with retrieved memory vectors.

    If the *retriever* exposes a ``retrieve`` method (as
    :class:`~memory.reinforce_retriever.ReinforceRetriever` does), the returned
    vector is assumed to already be weighted by retrieval probabilities which
    acts as a soft cross-attention over all candidate memories.  Otherwise the
    most recent message for ``speaker`` is encoded and added to the hidden
    state.
    """

    def __init__(
        self,
        retriever: Union[GraphRetriever, ReinforceRetriever],
        dim: int,
    ) -> None:
        self.retriever = retriever
        self.dim = dim

    def _encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        if not text:
            return vec
        for i, b in enumerate(text.encode("utf-8")):
            if i >= self.dim:
                break
            vec[i] = b / 255.0
        return vec

    def __call__(
        self, hidden_states: np.ndarray, dialogue_id: str, speaker: str
    ) -> np.ndarray:
        """Return ``hidden_states`` enriched with memory from the graph."""

        if hasattr(self.retriever, "retrieve"):
            mem_vec = self.retriever.retrieve(dialogue_id, speaker)
            if mem_vec is None:
                return hidden_states
            return hidden_states + mem_vec

        memory = self.retriever.last_message(dialogue_id, speaker)
        if not memory:
            return hidden_states
        mem_vec = self._encode(memory)
        return hidden_states + mem_vec


class QuantumHybridAttention:
    """Route patches through classical or quantum attention backends."""

    def __init__(self, router, quantum_backend) -> None:
        self.router = router
        self.quantum_backend = quantum_backend

    def _classical(self, query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
        scores = np.dot(query, key.T) / np.sqrt(key.shape[-1])
        weights = np.exp(scores)
        weights /= weights.sum(axis=-1, keepdims=True)
        return weights @ value

    def __call__(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        features: np.ndarray,
    ) -> np.ndarray:
        """Return attention outputs using a routing policy."""
        mask = self.router.route(features)
        out = np.zeros((query.shape[0], value.shape[-1]))
        if (~mask).any():
            out[~mask] = self._classical(query[~mask], key, value)
        if mask.any():
            for idx in np.where(mask)[0]:
                out[idx] = self.quantum_backend.attention(
                    query[idx], key, value
                )
        return out
