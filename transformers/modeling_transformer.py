"""Minimal transformer building blocks used in tests and demos.

This file introduces :class:`MemoryAttention`, a simple mechanism that
injects information retrieved from a memory graph into a sequence of hidden
states.  By default it works with :class:`~memory.storage.GraphRetriever`
but it can also consume a
:class:`~memory.reinforce_retriever.ReinforceRetriever` whose probability
distribution over nodes defines a soft cross-attention.  The goal is not to
implement a full Transformer model but to provide a lightweight hook where a
memory graph can influence the computation.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import ast

import numpy as np
import morphology
from .blocks.hyper_block import HyperBlock
from .resonant_layers import HarmonicResonanceLayer

from memory import GraphRetriever
from memory.reinforce_retriever import ReinforceRetriever
from .quantum_attention import QuantumAttention
from .quantum_memory_attention import QuantumMemoryAttention


_kernel: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None


def _phase_magnitude_add(*vecs: np.ndarray) -> np.ndarray:
    """Combine vectors preserving phase information.

    If none of the inputs are complex, this reduces to ordinary summation.
    Otherwise magnitudes and phases are summed separately and converted back
    into a complex vector.
    """

    if not any(np.iscomplexobj(v) for v in vecs):
        total = vecs[0]
        for v in vecs[1:]:
            total = total + v
        return total
    mags = sum(np.abs(v) for v in vecs)
    phase = None
    for v in vecs:
        if np.iscomplexobj(v) and np.any(v):
            phase = np.angle(v)
            break
    if phase is None:
        phase = 0.0
    return mags * np.exp(1j * phase)


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
        frequencies: Sequence[float] | None = None,
    ) -> None:
        self.retriever = retriever
        self.dim = dim
        if frequencies is None:
            frequencies = [1.0]
        self.resonance = HarmonicResonanceLayer(dim, np.asarray(frequencies, dtype=np.float32))

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

        if hasattr(self.retriever, "retrieve"):
            mem_vec = self.retriever.retrieve(dialogue_id, speaker)
            if mem_vec is not None:
                self.resonance.modulate(mem_vec)
            harmonic = self.resonance(hidden_states)
            if mem_vec is None:
                return _phase_magnitude_add(hidden_states, harmonic)
            if _kernel is not None:
                return _phase_magnitude_add(_kernel(hidden_states, mem_vec), harmonic)
            return _phase_magnitude_add(hidden_states, mem_vec, harmonic)

        memory = self.retriever.last_message(dialogue_id, speaker)
        if not memory:
            return _phase_magnitude_add(hidden_states, self.resonance(hidden_states))
        mem_vec = self._encode(memory)
        self.resonance.modulate(mem_vec)
        harmonic = self.resonance(hidden_states)
        if _kernel is not None:
            return _phase_magnitude_add(_kernel(hidden_states, mem_vec), harmonic)
        return _phase_magnitude_add(hidden_states, mem_vec, harmonic)


class QuantumHybridAttention:
    """Route patches through classical or quantum attention backends."""

    def __init__(self, router, quantum_backend) -> None:
        self.router = router
        self.quantum_backend = quantum_backend

    def _classical(
        self, query: np.ndarray, key: np.ndarray, value: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        scores = np.dot(query, key.T) / np.sqrt(key.shape[-1])
        magnitudes = np.abs(scores)
        phases = np.exp(1j * np.angle(scores))
        weights = np.exp(magnitudes)
        weights /= weights.sum(axis=-1, keepdims=True)
        out = (weights * phases) @ value
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

    def __call__(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        dialogue_id: str,
        speaker: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.attention.attention(
            query, key, value, dialogue_id, speaker
        )


class StaticBlock:
    """Linear block with fixed weights."""

    def __init__(self, weights: np.ndarray) -> None:
        self.weights = weights

    def __call__(
        self, x: np.ndarray, context: np.ndarray | None = None
    ) -> np.ndarray:
        return self.weights @ x


def block_factory(
    in_dim: int, out_dim: int, use_hyper: bool = False
) -> StaticBlock | HyperBlock:
    """Return a static or hyper block based on *use_hyper* flag."""
    if use_hyper:
        return HyperBlock(in_dim, out_dim)
    weights = np.eye(out_dim, in_dim, dtype=np.float32)
    return StaticBlock(weights)
