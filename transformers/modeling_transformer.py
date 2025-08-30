"""Minimal transformer building blocks used in tests and demos.

This file introduces :class:`MemoryAttention`, a simple mechanism that
injects information retrieved from :class:`~memory.memory_graph.GraphRetriever`
into a sequence of hidden states.  The goal is not to implement a full
Transformer model but to provide a lightweight hook where a memory graph
can influence the computation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from memory.memory_graph import GraphRetriever


class MemoryAttention:
    """Additively combines hidden states with retrieved memory vectors."""

    def __init__(self, retriever: GraphRetriever, dim: int) -> None:
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

        memory = self.retriever.last_message(dialogue_id, speaker)
        if not memory:
            return hidden_states
        mem_vec = self._encode(memory)
        return hidden_states + mem_vec
