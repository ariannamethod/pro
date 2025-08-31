import numpy as np
import pathlib
import sys

# Allow imports from project root
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from pro_predict import MiniSelfAttention  # noqa: E402


def test_logit_stochastic_with_fixed_seed():
    vocab = ["a", "b"]
    model = MiniSelfAttention(vocab, dim=4, use_gate=False)
    tokens = ["a"]
    np.random.seed(0)
    out1 = model.logits(tokens)
    np.random.seed(0)
    out2 = model.logits(tokens)
    assert any(out1[w] != out2[w] for w in vocab)
