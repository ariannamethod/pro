import os
import asyncio
import pro_engine
import pro_memory
import pro_rag
import pro_tune


def test_process_message_resilience(monkeypatch):
    for path in [pro_memory.DB_PATH, "pro_state.json", "dataset_sha.json"]:
        if os.path.exists(path):
            os.remove(path)
    engine = pro_engine.ProEngine()
    asyncio.run(engine.setup())

    orig_add = pro_memory.add_message
    orig_retrieve = pro_rag.retrieve

    async def failing_add_message(content: str) -> None:
        raise RuntimeError("db fail")

    async def failing_retrieve(words, limit=5):
        raise RuntimeError("rag fail")

    monkeypatch.setattr(pro_memory, "add_message", failing_add_message)
    monkeypatch.setattr(pro_rag, "retrieve", failing_retrieve)

    response, metrics = asyncio.run(engine.process_message("hello world"))
    assert isinstance(response, str)
    assert isinstance(metrics, dict)

    monkeypatch.setattr(pro_memory, "add_message", orig_add)
    monkeypatch.setattr(pro_rag, "retrieve", orig_retrieve)

    response2, metrics2 = asyncio.run(engine.process_message("second message"))
    assert isinstance(response2, str)
    assert isinstance(metrics2, dict)


def test_async_tune_resilience(monkeypatch):
    engine = pro_engine.ProEngine()

    def failing_train(state, path, weight):
        raise RuntimeError("tune fail")

    monkeypatch.setattr(pro_tune, "train_weighted", failing_train)

    asyncio.run(engine._async_tune(["dummy"]))


def test_meta_controller_triggers_tuning(monkeypatch):
    engine = pro_engine.ProEngine()
    # avoid launching background worker
    monkeypatch.setattr(engine, "_start_tune_worker", lambda: None)
    mc = engine.meta_controller

    for _ in range(mc.window):
        asyncio.run(mc.update({"perplexity": 1.0}))

    assert mc._baseline is not None

    while not engine.dataset_queue.empty():
        engine.dataset_queue.get_nowait()

    for _ in range(mc.window):
        asyncio.run(mc.update({"perplexity": 10.0}))

    assert engine.dataset_queue.get_nowait() is None
