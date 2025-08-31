import numpy as np

from transformers.modeling_transformer import MemoryAttention, register_kernel


class DummyRetriever:
    def __init__(self, dim: int):
        self.dim = dim

    def retrieve(self, dialogue_id, speaker):
        return np.ones(self.dim, dtype=np.float32)


def test_custom_memory_kernel_executes_and_resets():
    dim = 4
    hidden = np.ones(dim, dtype=np.float32)
    retriever = DummyRetriever(dim)
    register_kernel("lambda h, m: h * 0.5 + m")
    attn = MemoryAttention(retriever, dim=dim)
    out = attn(hidden, "d", "s")
    harmonic = attn.resonance(hidden)
    assert np.allclose(out, hidden * 0.5 + np.ones(dim) + harmonic)
    # Clean up so subsequent tests use default behaviour
    register_kernel(None)
