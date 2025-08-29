import asyncio
import sqlite3

import pro_engine
import pro_memory


def test_response_uses_trigram_prediction():
    engine = pro_engine.ProEngine()
    engine.state["trigram_counts"] = {
        ("hello", "WORLD"): {"foo": 2},
        ("WORLD", "foo"): {"bar": 3},
        ("foo", "bar"): {"baz": 4},
    }
    engine.state["word_counts"] = {"foo": 2, "bar": 3, "baz": 4}
    sentence = engine.respond(["hello", "WORLD"])
    assert sentence == "Hello WORLD foo bar baz."


def test_predict_next_word_fallback_to_bigram():
    engine = pro_engine.ProEngine()
    engine.state["bigram_counts"] = {"world": {"hello": 2}}
    engine.state["word_counts"] = {"hello": 2}
    assert engine.predict_next_word("x", "world") == "hello"


def test_preserves_first_word_capitalization():
    engine = pro_engine.ProEngine()
    engine.state["trigram_counts"] = {
        ("nasa", "launch"): {"window": 1},
        ("launch", "window"): {"opens": 1},
        ("window", "opens"): {"today": 1},
    }
    engine.state["word_counts"] = {
        "window": 1,
        "opens": 1,
        "today": 1,
    }
    sentence = engine.respond(["NASA", "launch"])
    assert sentence == "NASA launch window opens today."


def test_duplicate_responses_suppressed(tmp_path, monkeypatch):
    db_path = tmp_path / "mem.db"
    monkeypatch.setattr(pro_memory, "DB_PATH", str(db_path))
    asyncio.run(pro_memory.init_db())
    engine = pro_engine.ProEngine()
    engine.state["trigram_counts"] = {
        ("hello", "world"): {"foo": 1},
        ("world", "foo"): {"bar": 1},
        ("foo", "bar"): {"baz": 1},
    }
    engine.state["word_counts"] = {
        "hello": 5,
        "world": 4,
        "foo": 3,
        "bar": 2,
        "baz": 1,
    }
    first = engine.respond(["hello", "world"])
    second = engine.respond(["hello", "world"])
    assert first != second
    conn = sqlite3.connect(pro_memory.DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM responses")
    count = cur.fetchone()[0]
    conn.close()
    assert count == 2
