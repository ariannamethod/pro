import numpy as np
import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pro_rag_embedding
from transformers.modeling_transformer import MemoryAttention


class _DummyRetriever:
    def __init__(self, vec: np.ndarray) -> None:
        self.vec = vec

    def retrieve(self, dialogue_id: str, speaker: str) -> np.ndarray:  # noqa: D401 - simple stub
        return self.vec


@pytest.mark.asyncio
async def test_phase_preserved_through_cycle():
    text = "phase memory test"
    mem_vec = await pro_rag_embedding.embed_sentence(text)
    retriever = _DummyRetriever(mem_vec)
    attn = MemoryAttention(retriever, mem_vec.shape[0])
    hidden = np.zeros_like(mem_vec)
    out = attn(hidden, "dlg", "user")
    assert np.allclose(np.angle(out), np.angle(mem_vec))

