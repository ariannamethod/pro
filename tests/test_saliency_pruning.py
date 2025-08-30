import time

import pro_engine


def fake_attention(tokens):
    """Sleep in proportion to token count to simulate compute cost."""

    time.sleep(0.01 * len(tokens))
    return len(tokens)


def test_saliency_threshold_zero_no_change():
    engine = pro_engine.ProEngine(saliency_threshold=0.0)
    engine.state["word_counts"] = {
        "a": 100,
        "b": 90,
        "c": 80,
        "d": 1,
        "e": 1,
    }
    tokens = ["a", "b", "c", "d", "e"]
    filtered = engine._drop_low_saliency(tokens)
    assert filtered == tokens
    assert fake_attention(filtered) == fake_attention(tokens)


def test_saliency_threshold_speedup():
    engine = pro_engine.ProEngine(saliency_threshold=80.0)
    engine.state["word_counts"] = {
        "a": 100,
        "b": 90,
        "c": 80,
        "d": 1,
        "e": 1,
    }
    tokens = ["a", "b", "c", "d", "e"]

    start = time.perf_counter()
    fake_attention(tokens)
    full_time = time.perf_counter() - start

    pruned = engine._drop_low_saliency(tokens)
    assert len(pruned) < len(tokens)

    start = time.perf_counter()
    fake_attention(pruned)
    pruned_time = time.perf_counter() - start

    assert pruned_time < full_time
