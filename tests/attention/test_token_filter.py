import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from morphology import filter_by_tags
from transformers.blocks.attention import wave_attention


def _rand_complex(shape):
    return (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(np.complex64)


def test_wave_attention_token_filter_speed_quality():
    tokens = ["я", "иду", "домой", "быстро"]
    tags = ["PRON", "VERB", "NOUN", "ADV"]
    idx = filter_by_tags(tokens, tags, include={"VERB", "NOUN"})
    mask = np.zeros(len(tokens), dtype=bool)
    mask[idx] = True
    query = _rand_complex((len(tokens), 8))
    key = _rand_complex((len(tokens), 8))
    value = _rand_complex((len(tokens), 8))

    out_masked = wave_attention(query, key, value, mask=mask)
    out_manual = wave_attention(query, key[mask], value[mask])

    assert np.allclose(out_masked, out_manual)

    full_ops = query.shape[0] * key.shape[0] * query.shape[1]
    masked_ops = query.shape[0] * mask.sum() * query.shape[1]
    assert masked_ops < full_ops
