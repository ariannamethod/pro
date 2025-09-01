import os
import sys
import asyncio
import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pro_memory  # noqa: E402


@pytest.mark.asyncio
async def test_build_index_builds_index(tmp_path, monkeypatch):
    db_path = tmp_path / "mem.db"
    monkeypatch.setattr(pro_memory, "DB_PATH", str(db_path))
    pro_memory._STORE = None
    await pro_memory.init_db()

    emb = np.zeros(3, dtype=np.float32)
    for i in range(5):
        content = f"msg {i}"
        fp = pro_memory._fingerprint(content)
        await pro_memory.persist_embedding(content, emb, fingerprint=fp)

    await pro_memory.build_index(batch_size=1)

    assert pro_memory._STORE is not None
    assert pro_memory._STORE.embeddings.shape[0] == 5

    await pro_memory.close_db()
