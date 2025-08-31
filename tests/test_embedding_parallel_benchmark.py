import asyncio
import os
import sys
import time

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pro_rag_embedding import embed_sentence


@pytest.mark.asyncio
async def test_parallel_embedding_latency_under_five_seconds():
    texts = ["benchmark"] * 1000
    start = time.perf_counter()
    await asyncio.gather(*(embed_sentence(t) for t in texts))
    duration = time.perf_counter() - start
    assert duration < 5, f"parallel embedding took {duration:.2f}s"
