import asyncio
import pro_engine
import pro_memory
import pro_predict


def test_forbidden_words_replaced(tmp_path, monkeypatch):
    db_path = tmp_path / "mem.db"
    monkeypatch.setattr(pro_memory, "DB_PATH", str(db_path))
    asyncio.run(pro_memory.init_db())
    engine = pro_engine.ProEngine()
    engine.state["trigram_counts"] = {
        ("hi", "earth"): {"foo": 2},
        ("earth", "foo"): {"bar": 3},
        ("foo", "bar"): {"baz": 4},
    }
    engine.state["word_counts"] = {
        "hi": 2,
        "earth": 2,
        "foo": 3,
        "bar": 4,
        "baz": 5,
    }
    monkeypatch.setattr(pro_predict, "suggest", lambda w, topn=3: {"hello": ["hi"], "world": ["earth"]}.get(w, []))
    monkeypatch.setattr(pro_predict, "lookup_analogs", lambda w: None)
    monkeypatch.setattr(pro_predict, "_VECTORS", {
        "hi": {"a": 1.0},
        "earth": {"a": 0.9},
        "foo": {"a": 0.8},
        "bar": {"a": 0.7},
        "baz": {"a": 0.6},
        "qux": {"b": 1.0},
    })
    sentence = engine.respond(["hello", "world"], forbidden={"hello", "world"})
    assert "hello" not in sentence.lower()
    assert "world" not in sentence.lower()
    assert "hi" in sentence.lower()
    assert "earth" in sentence.lower()
