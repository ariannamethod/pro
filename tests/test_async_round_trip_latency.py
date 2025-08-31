import os
import sys
import time
import logging

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pro_memory  # noqa: E402
import pro_rag  # noqa: E402


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_round_trip_latency_benchmark():
    """Benchmark round-trip latency for database and vector-store
    operations."""
    if os.path.exists(pro_memory.DB_PATH):
        os.remove(pro_memory.DB_PATH)
    await pro_memory.init_db()

    db_latencies = []
    heavy_text = "benchmark data " * 50  # ~1000 chars simulates realistic payload
    for i in range(20):  # emulate 20 message exchanges
        start = time.perf_counter()
        msg = f"{heavy_text}{i}"
        await pro_memory.add_message(msg)
        await pro_memory.store_response(msg)
        await pro_memory.fetch_recent_messages()
        db_latencies.append(time.perf_counter() - start)

    await pro_memory.build_index()

    vector_latencies = []
    for i in range(20):
        start = time.perf_counter()
        await pro_rag.retrieve(["benchmark", "data", str(i)])
        vector_latencies.append(time.perf_counter() - start)

    all_latencies = db_latencies + vector_latencies
    avg_latency = sum(all_latencies) / len(all_latencies)

    logger.info("DB latencies: %s", [f"{lat:.4f}" for lat in db_latencies])
    logger.info(
        "Vector latencies: %s",
        [f"{lat:.4f}" for lat in vector_latencies],
    )
    logger.info(
        "Average round-trip latency: %.4fs over %d operations",
        avg_latency,
        len(all_latencies),
    )

    assert avg_latency < 5, (
        f"Average latency {avg_latency:.2f}s exceeds 5 seconds"
    )

    await pro_memory.close_db()
    if os.path.exists(pro_memory.DB_PATH):
        os.remove(pro_memory.DB_PATH)
