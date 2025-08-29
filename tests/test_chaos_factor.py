import random
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pro_engine
import pro_sequence


def test_inverse_frequency_and_chaos_unique():
    state = {
        'word_counts': {},
        'bigram_counts': {},
        'trigram_counts': {},
        'char_ngram_counts': {},
    }
    pro_sequence.analyze_sequences(state, ["foo", "bar", "foo"])
    assert state['word_inv']["foo"] == pytest.approx(0.5)
    assert state['word_inv']["bar"] == pytest.approx(1.0)

    engine = pro_engine.ProEngine()
    engine.state = state

    random.seed(0)
    seq0 = engine.plan_sentence([], 2, chaos_factor=0.0)
    assert seq0[0] == "foo"

    random.seed(0)
    seq1 = engine.plan_sentence([], 2, chaos_factor=1000.0)
    assert seq1[0] == "bar"
    assert len({w.lower() for w in seq1}) == len(seq1)
