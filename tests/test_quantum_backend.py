import importlib.util
import numpy as np
import pytest

from quantum.attention_backend import (
    AttentionBackend,
    QuantumAttentionBackend,
    create_attention_backend,
)


def test_quantum_attention_backend_shape():
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    backend = QuantumAttentionBackend(shots=16)
    q = np.ones((2, 4))
    k = np.ones((4, 4))
    v = np.arange(16).reshape(4, 4)
    out, betti = backend.attention(q[0], k, v)
    assert out.shape == (4,)
    assert betti.shape == (2,)


def test_backend_factory():
    backend = create_attention_backend("classical")
    assert isinstance(backend, AttentionBackend)
    q = np.ones((1, 4))
    k = np.ones((4, 4))
    v = np.arange(16).reshape(4, 4)
    out, betti = backend.attention(q[0], k, v)
    assert out.shape == (4,)
    assert betti.shape == (2,)


def test_backend_factory_requires_qiskit():
    if (
        importlib.util.find_spec("qiskit") is None
        or importlib.util.find_spec("qiskit_aer") is None
    ):
        with pytest.raises(RuntimeError):
            create_attention_backend("qiskit")
    else:
        backend = create_attention_backend("qiskit")
        assert isinstance(backend, QuantumAttentionBackend)
