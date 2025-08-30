import os
import sqlite3
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pro_memory  # noqa: E402


@pytest.mark.asyncio
async def test_store_and_fetch_similar_embeddings(tmp_path, monkeypatch):
    db_path = tmp_path / "mem.db"
    monkeypatch.setattr(pro_memory, "DB_PATH", str(db_path))
    await pro_memory.init_db()
    await pro_memory.build_index()
    await pro_memory.add_message("hello world")
    await pro_memory.add_message("goodbye world")
    results = await pro_memory.fetch_similar_messages("hello there", top_k=1)
    assert results == ["hello world"]
    # ensure embedding persisted as binary blob
    conn = sqlite3.connect(str(db_path))
    cur = conn.execute("SELECT embedding FROM messages")
    blobs = [np.frombuffer(row[0], dtype=np.float32) for row in cur.fetchall()]
    assert all(blob.size > 0 for blob in blobs)
    conn.close()
    await pro_memory.close_db()
