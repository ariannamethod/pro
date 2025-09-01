import numpy as np
from quantum.quant4 import compress_weights, decompress_weights


def test_quant4_roundtrip():
    w = np.random.randn(7, 5).astype(np.float32)
    packed, scale, shape = compress_weights(w)
    restored = decompress_weights(packed, scale, shape)
    assert restored.shape == w.shape
    assert np.allclose(restored, w, atol=scale)
