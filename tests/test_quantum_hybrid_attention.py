import numpy as np
import pytest

from quantum.attention_backend import QuantumAttentionBackend
from router.policy import PatchRoutingPolicy
from transformers.modeling_transformer import QuantumHybridAttention


def test_quantum_hybrid_attention_routes():
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    backend = QuantumAttentionBackend(shots=16)
    policy = PatchRoutingPolicy(dim=1, seed=0)
    attention = QuantumHybridAttention(policy, backend)
    q = np.ones((3, 2))
    k = np.ones((3, 2))
    v = np.ones((3, 2))
    features = np.ones((3, 1))
    out, betti = attention(q, k, v, features)
    assert out.shape == (3, 2)
    assert betti.shape == (3, 2)
    assert np.all(betti == 0)
