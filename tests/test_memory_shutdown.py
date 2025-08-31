import asyncio
from memory import pooling, storage
import pytest

def test_close_pool_sync_no_loop(tmp_path):
    db = tmp_path / "pool.db"
    asyncio.run(pooling.init_pool(str(db)))
    pooling._close_pool_sync()
    assert pooling._POOL == []

@pytest.mark.asyncio
async def test_close_pool_sync_running_loop(tmp_path):
    db = tmp_path / "pool_async.db"
    await pooling.init_pool(str(db))
    pooling._close_pool_sync()
    await asyncio.sleep(0)
    assert pooling._POOL == []

def test_close_db_sync_no_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", str(tmp_path / "mem.db"))
    asyncio.run(storage.init_db())
    storage._close_db_sync()
    assert pooling._POOL == []

@pytest.mark.asyncio
async def test_close_db_sync_running_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", str(tmp_path / "mem_async.db"))
    await storage.init_db()
    storage._close_db_sync()
    await asyncio.sleep(0)
    assert pooling._POOL == []
