import asyncio
import pro_engine
import pro_memory
import pro_predict
import pro_rag


def test_predict_learns_new_words(tmp_path, monkeypatch):
    db_path = tmp_path / "mem.db"
    monkeypatch.setattr(pro_memory, "DB_PATH", str(db_path))
    asyncio.run(pro_memory.init_db())

    engine = pro_engine.ProEngine()

    monkeypatch.setattr(
        pro_engine.ProEngine,
        "respond",
        lambda self, seeds, vocab=None, **kwargs: "music",
    )

    async def dummy_retrieve(words, limit=5):
        return []

    monkeypatch.setattr(pro_rag, "retrieve", dummy_retrieve)

    asyncio.run(engine.setup())

    new_word = "zzzzword"
    assert new_word not in pro_predict._VECTORS
    before = pro_predict.suggest(new_word)

    asyncio.run(engine.process_message(f"{new_word} music"))

    assert new_word in pro_predict._VECTORS
    after = pro_predict.suggest(new_word)
    assert after != before
    assert after
