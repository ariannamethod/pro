import numpy as np
import pathlib
import sys

# Allow imports from project root
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from transformers.quantum_attention import QuantumAttention  # noqa: E402


def test_betti_ring_structure():
    qa = QuantumAttention()
    amp = np.array(
        [[1, 1, 1],
         [1, 0, 1],
         [1, 1, 1]],
        dtype=complex,
    )
    v = np.eye(3)
    _, betti = qa.measure(amp, v)
    assert np.array_equal(betti, np.array([1, 1]))
