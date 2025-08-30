import asyncio
import os

import pro_engine
import pro_memory
import pro_predict
import pro_rag


def test_mini_transformer_logits():
    vocab = ["a", "b", "c"]
    logits = pro_predict.transformer_logits(["a", "b"], vocab)
    assert set(logits) == set(vocab)


def test_process_message_blends_transformer(tmp_path, monkeypatch):
    db_path = tmp_path / "mem.db"
    monkeypatch.setattr(pro_memory, "DB_PATH", str(db_path))
    asyncio.run(pro_memory.init_db())
    engine = pro_engine.ProEngine()
    engine.state["word_counts"] = {"hello": 1, "world": 1, "foo": 1, "baz": 1}
    engine.state["bigram_counts"] = {"world": {"foo": 1}}
    engine.state["trigram_counts"] = {("hello", "world"): {"foo": 1}}
    async def fake_retrieve(words):
        return []

    monkeypatch.setattr(pro_rag, "retrieve", fake_retrieve)

    called = {}

    def fake_transformer(tokens, vocab):
        called["tokens"] = list(tokens)
        return {"baz": 1.0}

    monkeypatch.setattr(pro_predict, "transformer_logits", fake_transformer)

    captured = {}

    async def fake_respond(seed_words, vocab=None, **kwargs):
        captured["seed_words"] = list(seed_words)
        return "ok"

    engine.respond = fake_respond
    asyncio.run(engine.process_message("hello world"))
    assert called["tokens"] == ["hello", "world"]
    assert "foo" in captured["seed_words"]
    assert "baz" in captured["seed_words"]


def test_logits_shift_after_training(tmp_path, monkeypatch):
    db_path = tmp_path / "mem.db"
    monkeypatch.setattr(pro_memory, "DB_PATH", str(db_path))
    asyncio.run(pro_memory.init_db())
    # ensure clean transformer state
    if os.path.exists("pro_transformer.npz"):
        os.remove("pro_transformer.npz")
    asyncio.run(pro_memory.add_message("hello world"))
    asyncio.run(pro_memory.store_response("foo"))
    vocab = ["hello", "world", "foo"]
    before = pro_predict.transformer_logits(["hello", "world"], vocab)["foo"]
    asyncio.run(pro_predict.update_transformer(vocab))
    after = pro_predict.transformer_logits(["hello", "world"], vocab)["foo"]
    assert after != before
