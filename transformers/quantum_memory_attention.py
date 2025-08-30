"""Quantum attention incorporating retrieved memory vectors."""
from __future__ import annotations

import numpy as np

from memory.reinforce_retriever import ReinforceRetriever
from .quantum_attention import QuantumAttention


class QuantumMemoryAttention:
    """Combine quantum amplitudes with a retrieved memory vector.

    The class first computes standard quantum amplitudes between ``query`` and
    ``key`` using :class:`QuantumAttention`.  A second set of amplitudes is
    derived from a memory vector obtained via
    :class:`~memory.reinforce_retriever.ReinforceRetriever.retrieve`.  The two
    complex amplitudes are added, strengthening phases when memory and key
    align.
    """

    def __init__(
        self, retriever: ReinforceRetriever, backend: QuantumAttention | None = None
    ) -> None:
        self.retriever = retriever
        self.backend = backend or QuantumAttention()

    def compute_amplitudes(
        self, query: np.ndarray, key: np.ndarray, dialogue_id: str, speaker: str
    ) -> np.ndarray:
        """Return amplitudes from keys and retrieved memory."""

        mem_vec = self.retriever.retrieve(dialogue_id, speaker)
        amp_key = self.backend.compute_amplitudes(query, key)
        amp_mem = self.backend.compute_amplitudes(query, mem_vec[None, :])
        return amp_key + amp_mem

    def attention(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        dialogue_id: str,
        speaker: str,
    ) -> np.ndarray:
        """Return attention output enriched with memory phases."""

        amp = self.compute_amplitudes(query, key, dialogue_id, speaker)
        return self.backend.measure(amp, value)
