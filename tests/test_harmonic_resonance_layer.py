import numpy as np

from transformers.modeling_transformer import MemoryAttention
from transformers.resonant_layers import HarmonicResonanceLayer


def test_shared_weights_across_layers() -> None:
    freqs = [0.1, 0.2]
    layer1 = HarmonicResonanceLayer(4, freqs)
    layer2 = HarmonicResonanceLayer(4, freqs)
    assert layer1.weights is layer2.weights
    layer1.modulate(np.ones(4, dtype=np.float32))
    assert np.allclose(layer1.weights, layer2.weights)


def test_memory_attention_forward_pass() -> None:
    class DummyRetriever:
        def last_message(self, dialogue_id, speaker):
            return None

    attn = MemoryAttention(DummyRetriever(), dim=4, frequencies=[0.1, 0.2])
    hidden = np.zeros(4, dtype=np.float32)
    out = attn(hidden, "d", "s")
    assert out.shape == hidden.shape
    assert not np.allclose(out, 0.0)
