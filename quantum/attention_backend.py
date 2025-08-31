"""Backends and factory for quantum-style attention mechanisms."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class AttentionBackend(Protocol):
    """Minimal protocol for attention backends."""

    def attention(
        self, query: np.ndarray, key: np.ndarray, value: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return attention output and optional Betti features."""


class QuantumAttentionBackend:
    """Evaluate attention weights using a quantum circuit simulation.

    This backend encodes the dot-product score between ``query`` and ``key`` as
    a rotation on a single qubit.  Measuring the qubit yields a probability that
    is used as an attention weight.  The circuit is executed on a small
    :class:`~qiskit.Aer` simulator which mimics NISQ hardware.
    """

    def __init__(self, shots: int = 256, backend_name: str = "aer_simulator") -> None:
        from qiskit import Aer, QuantumCircuit, execute  # type: ignore[import-not-found]

        self.shots = shots
        self.backend = Aer.get_backend(backend_name)
        self._qc = QuantumCircuit
        self._execute = execute

    def _prob(self, score: float) -> float:
        """Return probability of measuring |1> for a given score."""
        angle = float(np.tanh(score))  # map score to a bounded angle
        qc = self._qc(1, 1)
        qc.ry(2 * angle, 0)
        qc.measure(0, 0)
        job = self._execute(qc, backend=self.backend, shots=self.shots)
        counts = job.result().get_counts()
        return counts.get("1", 0) / self.shots

    def attention(
        self, query: np.ndarray, key: np.ndarray, value: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return attention output for ``query``/``key``/``value`` triples."""
        scores = np.dot(query, key.T) / np.sqrt(key.shape[-1])
        probs = np.array([self._prob(s) for s in scores])
        probs = probs / probs.sum() if probs.sum() else probs
        out = probs @ value
        betti = np.zeros(2, dtype=np.int64)
        return out, betti


def create_attention_backend(name: str, **kwargs) -> AttentionBackend:
    """Return an attention backend by *name*.

    Parameters
    ----------
    name:
        Backend identifier. ``"qiskit"`` uses :class:`QuantumAttentionBackend`
        while ``"classical"`` uses the pure NumPy implementation
        :class:`~transformers.quantum_attention.QuantumAttention`.
    kwargs:
        Passed through to the backend constructor.
    """

    if name == "qiskit":
        try:
            return QuantumAttentionBackend(**kwargs)
        except ImportError as exc:  # pragma: no cover - handled in tests
            raise RuntimeError(
                "qiskit backend selected but the 'qiskit' package is not installed"
            ) from exc
    if name == "classical":
        from transformers.quantum_attention import QuantumAttention

        return QuantumAttention(**kwargs)
    raise ValueError(f"Unknown attention backend: {name}")
