import asyncio
import sqlite3

import pro_engine
import pro_memory
import pro_metrics


def test_response_uses_trigram_prediction(tmp_path, monkeypatch):
    db_path = tmp_path / "mem.db"
    monkeypatch.setattr(pro_memory, "DB_PATH", str(db_path))
    asyncio.run(pro_memory.init_db())
    engine = pro_engine.ProEngine()
    engine.state["trigram_counts"] = {
        ("hello", "WORLD"): {"foo": 2},
        ("WORLD", "foo"): {"bar": 3},
        ("foo", "bar"): {"baz": 4},
    }
    engine.state["word_counts"] = {"foo": 2, "bar": 3, "baz": 4}
    sentence = engine.respond(["hello", "WORLD"])
    words = sentence[:-1].split()
    metrics = pro_metrics.compute_metrics(
        ["hello", "world"],
        engine.state["trigram_counts"],
        engine.state["bigram_counts"],
        engine.state["word_counts"],
        engine.state["char_ngram_counts"],
    )
    target = pro_metrics.target_length_from_metrics(metrics)
    assert words[:5] == ["Hello", "WORLD", "foo", "bar", "baz"]
    assert len(words) == target
    assert len(words) == len(set(words))


def test_predict_next_word_fallback_to_bigram():
    engine = pro_engine.ProEngine()
    engine.state["bigram_counts"] = {"world": {"hello": 2}}
    engine.state["word_counts"] = {"hello": 2}
    assert engine.predict_next_word("x", "world") == "hello"


def test_preserves_first_word_capitalization(tmp_path, monkeypatch):
    db_path = tmp_path / "mem.db"
    monkeypatch.setattr(pro_memory, "DB_PATH", str(db_path))
    asyncio.run(pro_memory.init_db())
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
    words = sentence[:-1].split()
    metrics = pro_metrics.compute_metrics(
        ["nasa", "launch"],
        engine.state["trigram_counts"],
        engine.state["bigram_counts"],
        engine.state["word_counts"],
        engine.state["char_ngram_counts"],
    )
    target = pro_metrics.target_length_from_metrics(metrics)
    assert words[:5] == ["NASA", "launch", "window", "opens", "today"]
    assert len(words) == target
    assert len(words) == len(set(words))


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
    words1 = first[:-1].split()
    words2 = second[:-1].split()
    metrics = pro_metrics.compute_metrics(
        ["hello", "world"],
        engine.state["trigram_counts"],
        engine.state["bigram_counts"],
        engine.state["word_counts"],
        engine.state["char_ngram_counts"],
    )
    target = pro_metrics.target_length_from_metrics(metrics)
    assert first != second
    assert len(words1) == target
    metrics2 = pro_metrics.compute_metrics(
        ["hello", "world", "alt0"],
        engine.state["trigram_counts"],
        engine.state["bigram_counts"],
        engine.state["word_counts"],
        engine.state["char_ngram_counts"],
    )
    target2 = pro_metrics.target_length_from_metrics(metrics2)
    assert len(words2) == target2
    assert len(words1) == len(set(words1))
    assert len(words2) == len(set(words2))
    conn = sqlite3.connect(pro_memory.DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM responses")
    count = cur.fetchone()[0]
    conn.close()
    assert count == 2


def test_response_variable_length_output(tmp_path, monkeypatch):
    db_path = tmp_path / "mem.db"
    monkeypatch.setattr(pro_memory, "DB_PATH", str(db_path))
    asyncio.run(pro_memory.init_db())
    engine = pro_engine.ProEngine()
    sentence1 = engine.respond(["a", "b"])
    sentence2 = engine.respond(["a", "b", "c", "d"])
    words1 = sentence1[:-1].split()
    words2 = sentence2[:-1].split()
    metrics1 = pro_metrics.compute_metrics(
        ["a", "b"],
        engine.state["trigram_counts"],
        engine.state["bigram_counts"],
        engine.state["word_counts"],
        engine.state["char_ngram_counts"],
    )
    metrics2 = pro_metrics.compute_metrics(
        ["a", "b", "c", "d"],
        engine.state["trigram_counts"],
        engine.state["bigram_counts"],
        engine.state["word_counts"],
        engine.state["char_ngram_counts"],
    )
    target1 = pro_metrics.target_length_from_metrics(metrics1)
    target2 = pro_metrics.target_length_from_metrics(metrics2)
    assert len(words1) == target1
    assert len(words2) == target2
    assert len(set(words1)) == len(words1)
    assert len(set(words2)) == len(words2)
    assert target1 != target2
