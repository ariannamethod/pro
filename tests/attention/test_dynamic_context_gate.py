import numpy as np
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from pro_predict import MiniSelfAttention


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
