from __future__ import annotations

"""Simple Qiskit-based backend to simulate attention on NISQ devices."""

import numpy as np
try:  # pragma: no cover - optional import for docs
    from qiskit import Aer, QuantumCircuit, execute
except Exception:  # pragma: no cover - allow tests without qiskit
    Aer = QuantumCircuit = execute = None  # type: ignore


class QuantumAttentionBackend:
    """Evaluate attention weights using a quantum circuit simulation.

    This backend encodes the dot-product score between ``query`` and ``key`` as
    a rotation on a single qubit.  Measuring the qubit yields a probability that
    is used as an attention weight.  The circuit is executed on a small
    :class:`~qiskit.Aer` simulator which mimics NISQ hardware.
    """

    def __init__(self, shots: int = 256, backend_name: str = "aer_simulator") -> None:
        self.shots = shots
        if Aer is not None:
            self.backend = Aer.get_backend(backend_name)
        else:  # pragma: no cover - qiskit not installed
            self.backend = None

    def _prob(self, score: float) -> float:
        """Return probability of measuring |1> for a given score."""
        if QuantumCircuit is None or self.backend is None:
            # Fall back to a classical sigmoid if qiskit is unavailable
            return float(1 / (1 + np.exp(-score)))

        angle = float(np.tanh(score))  # map score to a bounded angle
        qc = QuantumCircuit(1, 1)
        qc.ry(2 * angle, 0)
        qc.measure(0, 0)
        job = execute(qc, backend=self.backend, shots=self.shots)
        counts = job.result().get_counts()
        return counts.get("1", 0) / self.shots

    def attention(self, query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
        """Return attention output for ``query``/``key``/``value`` triples."""
        scores = np.dot(query, key.T) / np.sqrt(key.shape[-1])
        probs = np.array([self._prob(s) for s in scores])
        probs = probs / probs.sum() if probs.sum() else probs
        return probs @ value
