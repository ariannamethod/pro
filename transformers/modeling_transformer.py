"""Minimal transformer building blocks used in tests and demos.

This file introduces :class:`MemoryAttention`, a simple mechanism that
injects information retrieved from a memory graph into a sequence of hidden
states.  By default it works with :class:`~memory.memory_graph.GraphRetriever`
but it can also consume a
:class:`~memory.reinforce_retriever.ReinforceRetriever` whose probability
distribution over nodes defines a soft cross-attention.  The goal is not to
implement a full Transformer model but to provide a lightweight hook where a
memory graph can influence the computation.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import ast

import numpy as np
import morphology

from memory.memory_graph import GraphRetriever
from memory.reinforce_retriever import ReinforceRetriever
from .quantum_attention import QuantumAttention
from .quantum_memory_attention import QuantumMemoryAttention
from .blocks.attention import HoloMemoryGate


_kernel: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None


def register_kernel(fragment: Optional[str]) -> None:
    """Register a custom kernel fragment.

    The *fragment* must be a lambda expression of the form
    ``lambda h, m: ...``.  Pass ``None`` to remove an existing kernel.
    """

    global _kernel
    if fragment is None:
        _kernel = None
        return
    _kernel = _sanitize(fragment)


def _sanitize(fragment: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return a callable from *fragment* with a restricted environment."""

    tree = ast.parse(fragment, mode="eval")
    if not isinstance(tree.body, ast.Lambda):
        raise ValueError("Fragment must be a lambda expression")
    args = tree.body.args
    if [a.arg for a in args.args] != ["h", "m"]:
        raise ValueError("Lambda must have arguments (h, m)")

    allowed = (
        ast.Expression,
        ast.Lambda,
        ast.arguments,
        ast.arg,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Name,
        ast.Load,
        ast.Constant,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError("Disallowed syntax in fragment")
        if isinstance(node, ast.Name) and node.id not in {"h", "m"}:
            raise ValueError("Unknown identifier in fragment")

    return eval(compile(tree, "<memory_kernel>", "eval"), {"__builtins__": {}})


class ResonantAdapter:
    """Generate a sinusoidal adapter vector."""

    def __init__(self, frequency: float, amplitude: float) -> None:
        self.frequency = frequency
        self.amplitude = amplitude

    def __call__(self, dim: int) -> np.ndarray:
        idx = np.arange(dim, dtype=np.float32)
        return self.amplitude * np.sin(self.frequency * idx)


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
        self.gate = HoloMemoryGate(retriever, dim)

    def _encode(self, text: str) -> np.ndarray:
        """Encode ``text`` into a deterministic vector.

        The message is first converted into a base vector by normalising the
        raw bytes.  The word is then analysed morphologically using
        :func:`morphology.split` and the root and concatenated affixes are
        encoded into the first and second halves of the vector respectively.
        These sub-vectors are added to the base representation.
        """

        base = np.zeros(self.dim, dtype=np.float32)
        if not text:
            return base
        # Base encoding of the full word.
        for i, b in enumerate(text.encode("utf-8")):
            if i >= self.dim:
                break
            base[i] = b / 255.0

        # Morphological encoding.
        root, prefixes, suffixes = morphology.split(text)
        half = self.dim // 2
        root_vec = np.zeros(self.dim, dtype=np.float32)
        for i, b in enumerate(root.encode("utf-8")):
            if i >= half:
                break
            root_vec[i] = b / 255.0

        affixes = "".join(prefixes + suffixes)
        for i, b in enumerate(affixes.encode("utf-8")):
            if i >= self.dim - half:
                break
            root_vec[half + i] = b / 255.0

        return base + root_vec

    def __call__(
        self, hidden_states: np.ndarray, dialogue_id: str, speaker: str
    ) -> np.ndarray:
        """Return ``hidden_states`` enriched with memory from the graph."""
        adapter_vec = ResonantAdapter(1.0, 0.1)(hidden_states.shape[-1])
        holo = self.gate(dialogue_id).real.astype(hidden_states.dtype)
        hidden_states = hidden_states + holo

        if hasattr(self.retriever, "retrieve"):
            mem_vec = self.retriever.retrieve(dialogue_id, speaker)
            if mem_vec is None:
                return hidden_states + adapter_vec
            if _kernel is not None:
                return _kernel(hidden_states, mem_vec) + adapter_vec
            return hidden_states + mem_vec + adapter_vec

        memory = self.retriever.last_message(dialogue_id, speaker)
        if not memory:
            return hidden_states + adapter_vec
        mem_vec = self._encode(memory)
        if _kernel is not None:
            return _kernel(hidden_states, mem_vec) + adapter_vec
        return hidden_states + mem_vec + adapter_vec


class QuantumHybridAttention:
    """Route patches through classical or quantum attention backends."""

    def __init__(self, router, quantum_backend) -> None:
        self.router = router
        self.quantum_backend = quantum_backend

    def _classical(
        self, query: np.ndarray, key: np.ndarray, value: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        scores = np.dot(query, key.T) / np.sqrt(key.shape[-1])
        weights = np.exp(scores)
        weights /= weights.sum(axis=-1, keepdims=True)
        out = weights @ value
        betti = np.zeros((query.shape[0], 2), dtype=np.int64)
        return out, betti

    def __call__(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return attention outputs using a routing policy."""
        mask = self.router.route(features)
        out = np.zeros((query.shape[0], value.shape[-1]))
        betti = np.zeros((query.shape[0], 2), dtype=np.int64)
        if (~mask).any():
            classical_out, _ = self._classical(query[~mask], key, value)
            out[~mask] = classical_out
        if mask.any():
            for idx in np.where(mask)[0]:
                result = self.quantum_backend.attention(query[idx], key, value)
                if isinstance(result, tuple):
                    q_out, b = result
                else:  # Backends without betti features
                    q_out, b = result, np.zeros(2, dtype=np.int64)
                scale = 1.0 + b.sum()
                out[idx] = q_out * scale
                betti[idx] = b
        return out, betti


class QuantumMemoryLayer:
    """Integrate retrieved memory into a quantum attention step."""

    def __init__(
        self,
        retriever: ReinforceRetriever,
        backend: QuantumAttention | None = None,
    ) -> None:
        self.attention = QuantumMemoryAttention(retriever, backend)
        self.gate = HoloMemoryGate(retriever, retriever.dim)

    def __call__(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        dialogue_id: str,
        speaker: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        holo = self.gate(dialogue_id).real.astype(query.dtype)
        query = query + holo
        return self.attention.attention(
            query, key, value, dialogue_id, speaker
        )
