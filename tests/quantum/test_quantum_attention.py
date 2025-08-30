import numpy as np
import pathlib
import sys

# Allow imports from project root
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from transformers.quantum_attention import QuantumAttention  # noqa: E402
from transformers.quantum_memory_attention import (  # noqa: E402
    QuantumMemoryAttention,
)


def test_noise_robustness():
    """Small noise should not drastically alter measurement outcomes."""
    q = np.array([[1.0, 0.0]])
    k = np.eye(2)
    v = np.eye(2)
    clean = QuantumAttention(noise=0.0, seed=0)
    noisy = QuantumAttention(noise=0.2, seed=0)
    amp_clean = clean.compute_amplitudes(q, k)
    amp_noisy = noisy.compute_amplitudes(q, k)
    res_clean, betti_clean = clean.measure(amp_clean, v)
    res_noisy, betti_noisy = noisy.measure(amp_noisy, v)
    assert np.allclose(res_clean, res_noisy, atol=0.2)
    assert betti_clean.shape == (2,)
    assert betti_noisy.shape == (2,)


def test_superposition_learning():
    """Queries spanning keys should return a balanced superposed output."""
    q = np.array([[1.0, 1.0]])
    k = np.eye(2)
    v = np.eye(2)
    qa = QuantumAttention()
    amp = qa.compute_amplitudes(q, k)
    res, betti = qa.measure(amp, v)
    expected = np.array([[0.5, 0.5]])
    assert np.allclose(res, expected, atol=0.1)
    assert np.array_equal(betti, np.array([0, 0]))


class _DummyRetriever:
    def __init__(self, vec):
        self.vec = vec

    def retrieve(self, dialogue_id: str, speaker: str) -> np.ndarray:
        return self.vec


def test_memory_key_amplification():
    """Amplitudes strengthen when memory vector matches a key."""

    q = np.array([[1.0, 0.0]])
    k = np.array([[1.0, 0.0]])
    retriever = _DummyRetriever(k[0])
    qma = QuantumMemoryAttention(retriever)
    amp_combined = qma.compute_amplitudes(q, k, "d", "s")
    base_amp = QuantumAttention().compute_amplitudes(q, k)
    assert np.abs(amp_combined)[0, 0] > np.abs(base_amp)[0, 0]
