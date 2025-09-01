import os
import sys
import asyncio
import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pro_memory  # noqa: E402


@pytest.mark.asyncio
async def test_build_index_yields_control(tmp_path, monkeypatch):
    db_path = tmp_path / "mem.db"
    monkeypatch.setattr(pro_memory, "DB_PATH", str(db_path))
    pro_memory._STORE = None
    await pro_memory.init_db()

    emb = np.zeros(3, dtype=np.float32)
    for i in range(5):
        await pro_memory.persist_embedding(f"msg {i}", emb)

    calls = 0
    real_sleep = asyncio.sleep

    async def fake_sleep(delay, result=None):
        nonlocal calls
        calls += 1
        return await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    await pro_memory.build_index(batch_size=1, yield_every=2)
    assert calls == 2

    await pro_memory.close_db()
