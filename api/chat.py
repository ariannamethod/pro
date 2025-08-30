"""Very small chat API integrating the memory graph.

The API is purposely tiny – it merely demonstrates how dialogue turns are
serialized into :class:`~memory.memory_graph.MemoryGraphStore`.  Each call to
:func:`ChatAPI.process_message` stores the message and returns a numerical
representation that has been enriched by the :class:`~transformers.modeling_transformer.MemoryAttention`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from memory.memory_graph import GraphRetriever, MemoryGraphStore
from transformers.modeling_transformer import MemoryAttention


class ChatAPI:
    """Minimal interface for interacting with the memory graph."""

    def __init__(self, path: str = "memory_graph.json", dim: int = 32) -> None:
        self.store = MemoryGraphStore(path)
        self.retriever = GraphRetriever(self.store)
        self.attention = MemoryAttention(self.retriever, dim=dim)

    def process_message(
        self, dialogue_id: str, speaker: str, text: str
    ) -> np.ndarray:
        """Store a message and return its vector representation."""

        self.store.add_utterance(dialogue_id, speaker, text)
        hidden = np.zeros((1, self.attention.dim), dtype=np.float32)
        return self.attention(hidden, dialogue_id, speaker)
