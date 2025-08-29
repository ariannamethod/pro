import asyncio
import pytest
import pro_engine
import pro_memory
import pro_rag
import pro_tune


@pytest.mark.asyncio
async def test_process_message_resilience(monkeypatch):
    engine = pro_engine.ProEngine()
    await engine.setup()

    orig_add = pro_memory.add_message
    orig_retrieve = pro_rag.retrieve

    async def failing_add_message(content: str) -> None:
        raise RuntimeError("db fail")

    async def failing_retrieve(words, limit=5):
        raise RuntimeError("rag fail")

    monkeypatch.setattr(pro_memory, "add_message", failing_add_message)
    monkeypatch.setattr(pro_rag, "retrieve", failing_retrieve)

    response, metrics = await engine.process_message("hello world")
    assert isinstance(response, str)
    assert isinstance(metrics, dict)

    monkeypatch.setattr(pro_memory, "add_message", orig_add)
    monkeypatch.setattr(pro_rag, "retrieve", orig_retrieve)

    response2, metrics2 = await engine.process_message("second message")
    assert isinstance(response2, str)
    assert isinstance(metrics2, dict)


@pytest.mark.asyncio
async def test_async_tune_resilience(monkeypatch):
    engine = pro_engine.ProEngine()

    def failing_train(state, path):
        raise RuntimeError("tune fail")

    monkeypatch.setattr(pro_tune, "train", failing_train)

    await engine._async_tune()
