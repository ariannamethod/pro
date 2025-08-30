import os
import sys
import asyncio
import sqlite3

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pro_memory  # noqa: E402
import pro_rag  # noqa: E402
import pro_tune  # noqa: E402
from pro_engine import ProEngine  # noqa: E402


@pytest.fixture
def engine(monkeypatch):
    for path in [pro_memory.DB_PATH, "pro_state.json", "dataset_sha.json"]:
        if os.path.exists(path):
            os.remove(path)
    monkeypatch.setattr(pro_tune, "train", lambda *a, **k: None)
    eng = ProEngine()
    asyncio.run(eng.setup())
    return eng


def test_calls_memory_and_rag(engine, monkeypatch):
    add_calls = []
    retrieve_calls = []
    store_calls = []
    orig_add = pro_memory.add_message
    orig_retrieve = pro_rag.retrieve
    orig_store = pro_memory.store_response

    async def wrapped_add_message(content):
        add_calls.append(content)
        await orig_add(content)

    async def wrapped_retrieve(words, limit=5):
        retrieve_calls.append(list(words))
        return await orig_retrieve(words, limit)

    async def wrapped_store_response(content):
        store_calls.append(content)
        await orig_store(content)

    monkeypatch.setattr(pro_memory, "add_message", wrapped_add_message)
    monkeypatch.setattr(pro_rag, "retrieve", wrapped_retrieve)
    monkeypatch.setattr(pro_memory, "store_response", wrapped_store_response)

    asyncio.run(engine.process_message("hello world"))
    assert len(add_calls) == 2
    assert len(retrieve_calls) == 1
    assert len(store_calls) == 1


def test_sqlite_connection_open_close(engine, monkeypatch):
    counts = {"connect": 0, "close": 0}
    orig_connect = sqlite3.connect

    def tracking_connect(*args, **kwargs):
        counts["connect"] += 1
        conn = orig_connect(*args, **kwargs)

        class Wrapped:
            def __init__(self, conn):
                self._conn = conn

            def __getattr__(self, name):
                return getattr(self._conn, name)

            def close(self):
                counts["close"] += 1
                return self._conn.close()

        return Wrapped(conn)

    monkeypatch.setattr(sqlite3, "connect", tracking_connect)

    asyncio.run(engine.process_message("another message"))
    # With a pooled sqlite connection, no new connections should be opened.
    assert counts["connect"] == 0
    assert counts["close"] == 0
