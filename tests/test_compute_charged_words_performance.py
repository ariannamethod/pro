import os
import sys
import time
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pro_engine import ProEngine  # noqa: E402


def test_compute_charged_words_large_list_performance():
    engine = ProEngine()
    words = [f"word{i % 50}" for i in range(10000)]

    async def run() -> list[str]:
        return await asyncio.to_thread(engine.compute_charged_words, words)

    start = time.perf_counter()
    result = asyncio.run(run())
    elapsed = time.perf_counter() - start
    assert len(result) <= 5
    assert elapsed < 1.0
