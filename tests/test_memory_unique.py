import asyncio

import pro_memory


def test_is_unique_near_duplicate(tmp_path, monkeypatch):
    db_path = tmp_path / "mem.db"
    monkeypatch.setattr(pro_memory, "DB_PATH", str(db_path))
    asyncio.run(pro_memory.init_db())

    asyncio.run(pro_memory.store_response("hello world"))

    assert not asyncio.run(pro_memory.is_unique("hello world"))
    assert not asyncio.run(pro_memory.is_unique("Hello world!"))
    assert asyncio.run(pro_memory.is_unique("something else entirely"))
