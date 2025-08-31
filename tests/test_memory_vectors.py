import os
import sqlite3
import sys

import numpy as np
import pytest
from autograd import grad
from autograd import numpy as anp

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from memory import storage  # noqa: E402
import morphology  # noqa: E402


@pytest.mark.asyncio
async def test_store_and_fetch_similar_embeddings(tmp_path, monkeypatch):
    db_path = tmp_path / "mem.db"
    monkeypatch.setattr(storage, "DB_PATH", str(db_path))
    await storage.init_db()
    await storage.build_index()
    await storage.add_message("hello world")
    await storage.add_message("goodbye world")
    results = await storage.fetch_similar_messages("hello there", top_k=1)
    assert results == ["hello world"]
    # Update the first embedding and ensure it persists
    new_vec = np.ones_like(storage._STORE.embeddings[0])
    await storage.persist_learned_vector(0, new_vec)
    conn = sqlite3.connect(str(db_path))
    cur = conn.execute("SELECT embedding FROM messages WHERE rowid = 1")
    blob = np.frombuffer(cur.fetchone()[0], dtype=np.float32)
    assert np.allclose(blob, new_vec)
    conn.close()
    await storage.close_db()


def test_embedding_storage_and_gradient_retrieval():
    store = storage.MemoryStore()
    text = "привет мир"
    emb = store.add_utterance("dlg", "user", text)
    dialogue = store.get_dialogue("dlg")
    assert len(dialogue) == 1
    expected = morphology.encode(text)
    assert np.allclose(emb, expected)

    def loss_fn(q, mat):
        return store.retrieve(q, "dlg", "user", embeddings=mat) @ anp.ones(store.dim)

    query = anp.random.randn(store.dim)
    grad_q = grad(lambda q: loss_fn(q, store.embeddings))(query)
    grad_m = grad(lambda m: loss_fn(query, m))(store.embeddings)
    assert grad_q.shape == query.shape
    assert grad_m.shape == store.embeddings.shape
