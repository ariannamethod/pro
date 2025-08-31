import asyncio
import os
import pro_engine
import pro_memory
import pro_predict
import pro_rag


def test_predict_learns_new_words(tmp_path, monkeypatch):
    for path in ["datasets/conversation.log", "datasets/embeddings.pkl"]:
        if os.path.exists(path):
            os.remove(path)
    db_path = tmp_path / "mem.db"
    monkeypatch.setattr(pro_memory, "DB_PATH", str(db_path))
    asyncio.run(pro_memory.init_db())

    engine = pro_engine.ProEngine()

    async def fake_respond(self, seeds, vocab=None, **kwargs):
        return "music"

    monkeypatch.setattr(pro_engine.ProEngine, "respond", fake_respond)

    async def dummy_retrieve(words, limit=5):
        return []

    monkeypatch.setattr(pro_rag, "retrieve", dummy_retrieve)

    new_word = "zzzzword"
    assert new_word not in pro_predict._VECTORS
    before = asyncio.run(pro_predict.suggest_async(new_word))
    
    async def run_message():
        await engine.setup()
        await engine.process_message(f"{new_word} music")
        assert pro_predict.TOKENS_QUEUE is not None
        await pro_predict.TOKENS_QUEUE.join()
        assert pro_predict._SAVE_TASK is None
        await engine.shutdown()

    asyncio.run(run_message())

    assert new_word in pro_predict._VECTORS
    after = asyncio.run(pro_predict.suggest_async(new_word))
    assert after != before
    assert after
    assert pro_predict._SAVE_TASK is None
