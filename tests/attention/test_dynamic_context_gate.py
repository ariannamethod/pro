import numpy as np
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from pro_predict import MiniSelfAttention  # noqa: E402
from transformers.modeling_transformer import (  # noqa: E402
    MemoryAttention,
    ResonantAdapter,
)


def test_gate_can_zero_context():
    vocab = ["a", "b"]
    model = MiniSelfAttention(vocab, dim=4, use_gate=True)
    tokens = ["a"]
    baseline = model.logits(tokens)
    # Force gate to suppress context completely
    model.gate.bias = np.full(model.dim, -100.0)
    gated = model.logits(tokens)
    assert all(abs(v) < 1e-6 for v in gated.values())
    # With strong positive bias the logits should differ
    model.gate.bias = np.full(model.dim, 100.0)
    boosted = model.logits(tokens)
    assert any(abs(boosted[w]) > abs(baseline[w]) for w in vocab)


def test_resonant_adapter_adds_signal():
    class EmptyRetriever:
        def last_message(self, dialogue_id, speaker):
            return None

    attention = MemoryAttention(EmptyRetriever(), dim=4)
    hidden = np.zeros(4, dtype=np.float32)
    out = attention(hidden, "d", "s")
    expected = ResonantAdapter(1.0, 0.1)(4)
    assert np.allclose(out, expected)
