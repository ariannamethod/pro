import asyncio
import math

import pro_engine
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


def test_prepare_analog_map(monkeypatch):
    engine = pro_engine.ProEngine()

    async def fake_suggest(tok, topn=1):
        return []

    def fake_lookup(tok):
        return tok + "_analog"

    monkeypatch.setattr(pro_predict, "suggest_async", fake_suggest)
    monkeypatch.setattr(pro_predict, "lookup_analogs", fake_lookup)

    analog_map = asyncio.run(engine._prepare_analog_map({"foo"}))
    assert analog_map == {"foo": "foo_analog"}


def test_compute_target_length():
    engine = pro_engine.ProEngine()
    engine.state["trigram_counts"] = {}
    engine.state["bigram_counts"] = {}
    engine.state["word_counts"] = {"hello": 1, "world": 1}
    engine.state["char_ngram_counts"] = {}
    metrics, length = engine._compute_target_length(["hello", "world"])
    expected = pro_metrics.compute_metrics(
        ["hello", "world"],
        engine.state["trigram_counts"],
        engine.state["bigram_counts"],
        engine.state["word_counts"],
        engine.state["char_ngram_counts"],
    )
    expected_len = pro_metrics.target_length_from_metrics(expected)
    assert metrics == expected
    assert length == expected_len


def test_build_first_phrase(monkeypatch):
    engine = pro_engine.ProEngine()
    engine.state["trigram_counts"] = {
        ("hello", "WORLD"): {"foo": 2},
        ("WORLD", "foo"): {"bar": 3},
        ("foo", "bar"): {"baz": 4},
    }
    engine.state["word_counts"] = {"foo": 2, "bar": 3, "baz": 4}
    target = engine._compute_target_length(["hello", "WORLD"])[1]
    sentence1, first_words, _ = asyncio.run(
        engine._build_first_phrase(
            ["hello", "WORLD"],
            target,
            set(),
            0.0,
            {},
            {},
        )
    )
    words = sentence1.rstrip(".").split()
    assert words[:5] == ["Hello", "WORLD", "foo", "bar", "baz"]
    assert len(words) == target
    assert len(words) == len(set(words))
    assert first_words == [w.lower() for w in words]


def test_build_second_phrase(monkeypatch):
    engine = pro_engine.ProEngine()
    engine.state["trigram_counts"] = {
        ("hello", "WORLD"): {"foo": 2},
        ("WORLD", "foo"): {"bar": 3},
        ("foo", "bar"): {"baz": 4},
    }
    engine.state["word_counts"] = {"foo": 2, "bar": 3, "baz": 4}
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

    async def _noop():
        return None

    monkeypatch.setattr(pro_predict, "_ensure_vectors", _noop)

    target = engine._compute_target_length(["hello", "WORLD"])[1]
    sentence1, first_words, ordered = asyncio.run(
        engine._build_first_phrase(
            ["hello", "WORLD"],
            target,
            set(),
            0.0,
            {},
            {},
        )
    )
    sentence2 = asyncio.run(
        engine._build_second_phrase(
            first_words,
            ordered,
            0.3,
            set(),
            0.0,
        )
    )
    second_words = sentence2.rstrip(".").split()
    length2 = engine._compute_target_length(
        first_words, min_len=5, max_len=6, keys=["entropy", "perplexity"]
    )[1]
    assert len(second_words) == length2
    assert len(second_words) == len(set(second_words))
    assert set(first_words).isdisjoint({w.lower() for w in second_words})
    vecs = pro_predict._VECTORS
    sims = []
    for w2 in [w.lower() for w in second_words if w.lower() in vecs]:
        for w1 in [w.lower() for w in first_words if w.lower() in vecs]:
            sims.append(_cos(vecs[w2], vecs[w1]))
    assert sims and max(sims) < 0.1
