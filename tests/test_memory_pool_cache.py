import os
import sys
import asyncio
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pro_memory_pool  # noqa: E402

@pytest.mark.asyncio
async def test_execute_cached_ttl(tmp_path):
    db = tmp_path / "cache.db"
    await pro_memory_pool.init_pool(str(db))
    async with pro_memory_pool.get_connection() as conn:
        await conn.execute("INSERT INTO messages(content) VALUES (?)", ("hi",))
        await conn.commit()
    rows1 = await pro_memory_pool.execute_cached(
        "SELECT content FROM messages", ttl=0.5
    )
    async with pro_memory_pool.get_connection() as conn:
        await conn.execute("INSERT INTO messages(content) VALUES (?)", ("bye",))
        await conn.commit()
    rows2 = await pro_memory_pool.execute_cached(
        "SELECT content FROM messages", ttl=0.5
    )
    assert rows1 == rows2  # cached result
    await asyncio.sleep(0.6)
    rows3 = await pro_memory_pool.execute_cached(
        "SELECT content FROM messages", ttl=0.1
    )
    assert len(rows3) == 2
    await pro_memory_pool.close_pool()
