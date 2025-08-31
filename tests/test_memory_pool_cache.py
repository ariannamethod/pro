import os
import sys
import asyncio
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from memory import pooling  # noqa: E402

@pytest.mark.asyncio
async def test_execute_cached_ttl(tmp_path):
    db = tmp_path / "cache.db"
    await pooling.init_pool(str(db))
    async with pooling.get_connection() as conn:
        await asyncio.to_thread(
            conn.execute, "INSERT INTO messages(content) VALUES (?)", ("hi",)
        )
        await asyncio.to_thread(conn.commit)
    rows1 = await pooling.execute_cached(
        "SELECT content FROM messages", ttl=0.5
    )
    async with pooling.get_connection() as conn:
        await asyncio.to_thread(
            conn.execute, "INSERT INTO messages(content) VALUES (?)", ("bye",)
        )
        await asyncio.to_thread(conn.commit)
    rows2 = await pooling.execute_cached(
        "SELECT content FROM messages", ttl=0.5
    )
    assert rows1 == rows2  # cached result
    await asyncio.sleep(0.6)
    rows3 = await pooling.execute_cached(
        "SELECT content FROM messages", ttl=0.1
    )
    assert len(rows3) == 2
    await pooling.close_pool()
