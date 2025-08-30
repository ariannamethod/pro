import numpy as np

from quantum.attention_backend import QuantumAttentionBackend


def test_quantum_attention_backend_shape():
    backend = QuantumAttentionBackend(shots=16)
    q = np.ones((2, 4))
    k = np.ones((4, 4))
    v = np.arange(16).reshape(4, 4)
    out = backend.attention(q[0], k, v)
    assert out.shape == (4,)
