import asyncio
import sqlite3
import random

import pro_engine
import pro_memory
import pro_predict


def test_response_uses_trigram_prediction():
    engine = pro_engine.ProEngine()
    engine.state["trigram_counts"] = {
        ("hello", "WORLD"): {"foo": 2},
        ("WORLD", "foo"): {"bar": 3},
        ("foo", "bar"): {"baz": 4},
    }
    engine.state["word_counts"] = {"foo": 2, "bar": 3, "baz": 4}
    sentence = engine.respond(["hello", "WORLD"])
    first_sentence = sentence.split(".")[0] + "."
    assert first_sentence == "Hello WORLD foo bar baz."


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
    first_sentence = sentence.split(".")[0] + "."
    assert first_sentence == "NASA launch window opens today."


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


def _stub_dissimilar(words, count):
    return [f"w{i}" for i in range(count)]


def test_second_sentence_length(monkeypatch):
    engine = pro_engine.ProEngine()
    engine.state["trigram_counts"] = {
        ("a", "b"): {"c": 1},
        ("b", "c"): {"d": 1},
        ("c", "d"): {"e": 1},
    }
    engine.state["word_counts"] = {"c": 1, "d": 1, "e": 1}
    monkeypatch.setattr(pro_memory, "is_unique", lambda x: True)
    monkeypatch.setattr(pro_memory, "store_response", lambda x: None)
    monkeypatch.setattr(pro_predict, "dissimilar", _stub_dissimilar)
    monkeypatch.setattr(random, "choice", lambda seq: 5)
    resp = engine.respond(["a", "b"])
    second_words = resp.split(".")[1].strip().split()
    assert len(second_words) == 5
    monkeypatch.setattr(random, "choice", lambda seq: 6)
    resp = engine.respond(["a", "b"])
    second_words = resp.split(".")[1].strip().split()
    assert len(second_words) == 6


def test_no_shared_words_between_sentences(monkeypatch):
    engine = pro_engine.ProEngine()
    engine.state["trigram_counts"] = {
        ("hello", "world"): {"foo": 1},
        ("world", "foo"): {"bar": 1},
        ("foo", "bar"): {"baz": 1},
    }
    engine.state["word_counts"] = {"foo": 1, "bar": 1, "baz": 1}
    monkeypatch.setattr(pro_memory, "is_unique", lambda x: True)
    monkeypatch.setattr(pro_memory, "store_response", lambda x: None)
    monkeypatch.setattr(pro_predict, "dissimilar", _stub_dissimilar)
    monkeypatch.setattr(random, "choice", lambda seq: 5)
    resp = engine.respond(["hello", "world"])
    parts = resp.split(".")
    first_words = set(parts[0].lower().split())
    second_words = set(parts[1].strip().lower().split())
    assert first_words.isdisjoint(second_words)
