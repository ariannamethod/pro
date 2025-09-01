import numpy as np
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from pro_predict import MiniSelfAttention  # noqa: E402
from transformers.blocks import ResonantDropout  # noqa: E402
from transformers.modeling_transformer import MemoryAttention  # noqa: E402


def test_gate_can_zero_context():
    vocab = ["a", "b"]
    model = MiniSelfAttention(vocab, dim=4, use_gate=True)
    tokens = ["a"]
    baseline = model.logits(tokens)
    # Force gate to suppress context completely
    model.gate.bias = np.full(model.dim, -100.0)
    gated = model.logits(tokens)
    expected = np.full(len(vocab), 1 / len(vocab))
    np.testing.assert_allclose([gated[w] for w in vocab], expected, rtol=1e-2)
    # With strong positive bias the logits should differ
    model.gate.bias = np.full(model.dim, 100.0)
    boosted = model.logits(tokens)
    assert any(abs(boosted[w]) > abs(baseline[w]) for w in vocab)


def test_harmonic_layer_adds_signal():
    class EmptyRetriever:
        def last_message(self, dialogue_id, speaker):
            return None

    attention = MemoryAttention(EmptyRetriever(), dim=4)
    hidden = np.zeros(4, dtype=np.float32)
    out = attention(hidden, "d", "s")
    expected = attention.resonance(hidden)
    assert np.allclose(out, hidden + expected)


def test_resonant_dropout_depends_on_frequency():
    ctx = np.ones((10, 100), dtype=np.float32)
    slow = ResonantDropout(0.1, seed=0)
    fast = ResonantDropout(1.0, seed=0)
    out_slow = slow(ctx)
    out_fast = fast(ctx)
    zero_slow = np.mean(out_slow == 0.0)
    zero_fast = np.mean(out_fast == 0.0)
    assert zero_slow != zero_fast
