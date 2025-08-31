import asyncio
import pro_memory
import pro_memory_pool
import pytest


def test_close_pool_sync_no_loop(tmp_path):
    db = tmp_path / "pool.db"
    asyncio.run(pro_memory_pool.init_pool(str(db)))
    pro_memory_pool._close_pool_sync()
    assert pro_memory_pool._POOL == []


@pytest.mark.asyncio
async def test_close_pool_sync_running_loop(tmp_path):
    db = tmp_path / "pool_async.db"
    await pro_memory_pool.init_pool(str(db))
    pro_memory_pool._close_pool_sync()
    await asyncio.sleep(0)
    assert pro_memory_pool._POOL == []


def test_close_db_sync_no_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(pro_memory, "DB_PATH", str(tmp_path / "mem.db"))
    asyncio.run(pro_memory.init_db())
    pro_memory._close_db_sync()
    assert pro_memory_pool._POOL == []


@pytest.mark.asyncio
async def test_close_db_sync_running_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(pro_memory, "DB_PATH", str(tmp_path / "mem_async.db"))
    await pro_memory.init_db()
    pro_memory._close_db_sync()
    await asyncio.sleep(0)
    assert pro_memory_pool._POOL == []

