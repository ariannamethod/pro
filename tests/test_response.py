import asyncio
import math
import os
import sqlite3

import pro_engine
import pro_memory
import pro_metrics
import pro_predict


def _cos(a, b):
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _split_two_sentences(text):
    first, second = text.split(". ")
    return first.split(), second.rstrip(".").split()


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
    engine.state["word_counts"] = {
        "foo": 2,
        "bar": 3,
        "baz": 4,
    }
    monkeypatch.setattr(
        pro_predict,
        "_VECTORS",
        {
            "foo": {"a": 1.0},
            "bar": {"a": 0.8},
            "baz": {"a": 0.6},
            "qux": {"b": 1.0},
            "quux": {"b": 0.8},
            "corge": {"b": 0.6},
        },
    )
    real_exists = os.path.exists
    monkeypatch.setattr(
        os.path,
        "exists",
        lambda p, _real=real_exists: False if p == "datasets" else _real(p),
    )
    sentence = engine.respond(["hello", "WORLD"])
    first_words, second_words = _split_two_sentences(sentence)
    metrics = pro_metrics.compute_metrics(
        ["hello", "world"],
        engine.state["trigram_counts"],
        engine.state["bigram_counts"],
        engine.state["word_counts"],
        engine.state["char_ngram_counts"],
    )
    target = pro_metrics.target_length_from_metrics(metrics)
    assert first_words[:5] == ["Hello", "WORLD", "foo", "bar", "baz"]
    assert len(first_words) == target
    assert len(first_words) == len(set(first_words))
    metrics_first = pro_metrics.compute_metrics(
        [w.lower() for w in first_words],
        engine.state["trigram_counts"],
        engine.state["bigram_counts"],
        engine.state["word_counts"],
        engine.state["char_ngram_counts"],
    )
    target2 = pro_metrics.target_length_from_metrics(
        {
            "entropy": metrics_first["entropy"],
            "perplexity": metrics_first["perplexity"],
        },
        min_len=5,
        max_len=6,
    )
    assert len(second_words) == target2
    assert len(second_words) == len(set(second_words))
    assert set(w.lower() for w in first_words).isdisjoint(
        set(w.lower() for w in second_words)
    )
    vecs = pro_predict._VECTORS
    sims = []
    for w2 in [w.lower() for w in second_words if w.lower() in vecs]:
        for w1 in [w.lower() for w in first_words if w.lower() in vecs]:
            sims.append(_cos(vecs[w2], vecs[w1]))
    assert sims and max(sims) < 0.1


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
    first_words, second_words = _split_two_sentences(sentence)
    metrics = pro_metrics.compute_metrics(
        ["nasa", "launch"],
        engine.state["trigram_counts"],
        engine.state["bigram_counts"],
        engine.state["word_counts"],
        engine.state["char_ngram_counts"],
    )
    target = pro_metrics.target_length_from_metrics(metrics)
    assert first_words[:5] == ["NASA", "launch", "window", "opens", "today"]
    assert len(first_words) == target
    assert len(first_words) == len(set(first_words))
    metrics_first = pro_metrics.compute_metrics(
        [w.lower() for w in first_words],
        engine.state["trigram_counts"],
        engine.state["bigram_counts"],
        engine.state["word_counts"],
        engine.state["char_ngram_counts"],
    )
    target2 = pro_metrics.target_length_from_metrics(
        {
            "entropy": metrics_first["entropy"],
            "perplexity": metrics_first["perplexity"],
        },
        min_len=5,
        max_len=6,
    )
    assert len(second_words) == target2
    assert len(second_words) == len(set(second_words))
    assert set(w.lower() for w in first_words).isdisjoint(
        set(w.lower() for w in second_words)
    )


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
    f1_first, f1_second = _split_two_sentences(first)
    f2_first, f2_second = _split_two_sentences(second)
    metrics = pro_metrics.compute_metrics(
        ["hello", "world"],
        engine.state["trigram_counts"],
        engine.state["bigram_counts"],
        engine.state["word_counts"],
        engine.state["char_ngram_counts"],
    )
    target = pro_metrics.target_length_from_metrics(metrics)
    assert first != second
    assert len(f1_first) == target
    metrics2 = pro_metrics.compute_metrics(
        ["hello", "world", "alt0"],
        engine.state["trigram_counts"],
        engine.state["bigram_counts"],
        engine.state["word_counts"],
        engine.state["char_ngram_counts"],
    )
    target2 = pro_metrics.target_length_from_metrics(metrics2)
    assert len(f2_first) == target2
    metrics_first = pro_metrics.compute_metrics(
        [w.lower() for w in f1_first],
        engine.state["trigram_counts"],
        engine.state["bigram_counts"],
        engine.state["word_counts"],
        engine.state["char_ngram_counts"],
    )
    metrics_second = pro_metrics.compute_metrics(
        [w.lower() for w in f2_first],
        engine.state["trigram_counts"],
        engine.state["bigram_counts"],
        engine.state["word_counts"],
        engine.state["char_ngram_counts"],
    )
    target_f1_second = pro_metrics.target_length_from_metrics(
        {
            "entropy": metrics_first["entropy"],
            "perplexity": metrics_first["perplexity"],
        },
        min_len=5,
        max_len=6,
    )
    target_f2_second = pro_metrics.target_length_from_metrics(
        {
            "entropy": metrics_second["entropy"],
            "perplexity": metrics_second["perplexity"],
        },
        min_len=5,
        max_len=6,
    )
    assert len(f1_second) == target_f1_second
    assert len(f2_second) == target_f2_second
    assert len(f1_first) == len(set(f1_first))
    assert len(f2_first) == len(set(f2_first))
    assert len(f1_second) == len(set(f1_second))
    assert len(f2_second) == len(set(f2_second))
    assert set(w.lower() for w in f1_first).isdisjoint(
        set(w.lower() for w in f1_second)
    )
    assert set(w.lower() for w in f2_first).isdisjoint(
        set(w.lower() for w in f2_second)
    )
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
    f1_first, f1_second = _split_two_sentences(sentence1)
    f2_first, f2_second = _split_two_sentences(sentence2)
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
    assert len(f1_first) == target1
    assert len(f2_first) == target2
    metrics_f1 = pro_metrics.compute_metrics(
        [w.lower() for w in f1_first],
        engine.state["trigram_counts"],
        engine.state["bigram_counts"],
        engine.state["word_counts"],
        engine.state["char_ngram_counts"],
    )
    metrics_f2 = pro_metrics.compute_metrics(
        [w.lower() for w in f2_first],
        engine.state["trigram_counts"],
        engine.state["bigram_counts"],
        engine.state["word_counts"],
        engine.state["char_ngram_counts"],
    )
    target1b = pro_metrics.target_length_from_metrics(
        {
            "entropy": metrics_f1["entropy"],
            "perplexity": metrics_f1["perplexity"],
        },
        min_len=5,
        max_len=6,
    )
    target2b = pro_metrics.target_length_from_metrics(
        {
            "entropy": metrics_f2["entropy"],
            "perplexity": metrics_f2["perplexity"],
        },
        min_len=5,
        max_len=6,
    )
    assert len(f1_second) == target1b
    assert len(f2_second) == target2b
    assert len(set(f1_first)) == len(f1_first)
    assert len(set(f2_first)) == len(f2_first)
    assert len(set(f1_second)) == len(f1_second)
    assert len(set(f2_second)) == len(f2_second)
    assert set(w.lower() for w in f1_first).isdisjoint(
        set(w.lower() for w in f1_second)
    )
    assert set(w.lower() for w in f2_first).isdisjoint(
        set(w.lower() for w in f2_second)
    )
    assert target1 != target2
