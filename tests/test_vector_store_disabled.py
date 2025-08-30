import importlib
import pytest

@pytest.mark.asyncio
async def test_vector_store_noop_when_url_missing(monkeypatch):
    monkeypatch.delenv("VECTOR_STORE_URL", raising=False)
    from api import vector_store as vs
    importlib.reload(vs)

    await vs.upsert("hello", [0.1, 0.2])  # should not raise
    assert await vs.query([0.1, 0.2]) == []
