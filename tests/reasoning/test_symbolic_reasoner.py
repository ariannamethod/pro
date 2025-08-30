import numpy as np
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from transformers.blocks import SymbolicReasoner, SymbolicAnd, SymbolicOr, SymbolicNot


def test_boolean_layers():
    a = np.array([1, 0], dtype=np.float32)
    b = np.array([0, 1], dtype=np.float32)
    and_layer = SymbolicAnd()
    or_layer = SymbolicOr()
    not_layer = SymbolicNot()

    assert and_layer(a, b).tolist() == [0.0, 0.0]
    assert or_layer(a, b).tolist() == [1.0, 1.0]
    assert not_layer(a).tolist() == [0.0, 1.0]


def test_reasoner_eval():
    reasoner = SymbolicReasoner()
    facts = {"A": True, "B": False}
    assert reasoner.evaluate("A AND NOT B", facts) is True
    assert reasoner.evaluate("A OR B", facts) is True
    assert reasoner.evaluate("NOT A", facts) is False
