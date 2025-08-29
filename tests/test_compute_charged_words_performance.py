import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pro_engine import ProEngine  # noqa: E402


def test_compute_charged_words_large_list_performance():
    engine = ProEngine()
    words = [f"word{i % 50}" for i in range(10000)]
    start = time.perf_counter()
    result = engine.compute_charged_words(words)
    elapsed = time.perf_counter() - start
    assert len(result) <= 5
    assert elapsed < 1.0
