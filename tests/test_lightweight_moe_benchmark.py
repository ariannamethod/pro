import pathlib
import sys
import time

import numpy as np

# Ensure local packages are importable
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from transformers.blocks import HyperBlock, LightweightMoEBlock


def test_lightweight_moe_benchmark():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16, dtype=np.float32)
    context = rng.standard_normal(16, dtype=np.float32)

    hyper = HyperBlock(16, 16)
    moe = LightweightMoEBlock(16, num_experts=4, seed=0)

    n = 200

    start = time.perf_counter()
    for _ in range(n):
        hyper(x, context)
    hyper_duration = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(n):
        moe(x, adapters=[{"bias": 0.1}])
    moe_duration = time.perf_counter() - start

    hyper_throughput = n / hyper_duration
    moe_throughput = n / moe_duration

    # Basic sanity checks so the benchmark executes
    assert hyper_duration > 0 and moe_duration > 0
    assert hyper_throughput > 0 and moe_throughput > 0
    # Ensure MoE isn't catastrophically slower than HyperBlock
    assert moe_duration < hyper_duration * 50
