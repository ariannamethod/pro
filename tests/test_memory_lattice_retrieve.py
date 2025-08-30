import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pro_metrics import tokenize, lowercase  # noqa: E402
import pro_memory  # noqa: E402
import pro_rag  # noqa: E402
from memory.memory_lattice import MemoryGraphStore  # noqa: E402
import pro_predict  # noqa: E402


def _embed(text: str) -> dict:
    words = lowercase(tokenize(text))
    return pro_rag._sentence_vector(words)


@pytest.mark.asyncio
async def test_retrieve_uses_graph_and_embedding(monkeypatch):
    monkeypatch.setattr(
        pro_predict, "_VECTORS", {"dogs": {"d": 1.0}, "cats": {"c": 1.0}}
    )
    store = MemoryGraphStore()
    store.add_utterance("d1", "user", "cats eat fish", _embed("cats eat fish"))
    store.add_utterance(
        "d1", "user", "dogs chase cats", _embed("dogs chase cats")
    )

    async def fake_related(words):
        return []

    monkeypatch.setattr(pro_memory, "fetch_related_concepts", fake_related)

    results = await pro_rag.retrieve(["dogs"], lattice=store, limit=2)
    assert results[0] == "dogs chase cats"
    assert "cats eat fish" in results
