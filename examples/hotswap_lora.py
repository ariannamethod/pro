"""Demonstrate HotSwapLoRAAdapter and measure throughput."""
from __future__ import annotations

import json
import time
import numpy as np

from transformers.modeling_transformer import HotSwapLoRAAdapter


class _DummyModel:
    """Replicates the minimal model used by the API."""

    def __init__(self, dim: int = 32) -> None:
        self.attention = np.zeros((dim, dim), dtype=np.float32)
        self.ffn = np.zeros((dim, dim), dtype=np.float32)


def _benchmark(model: _DummyModel, steps: int = 1000) -> float:
    x = np.zeros(model.attention.shape[0], dtype=np.float32)
    start = time.time()
    for _ in range(steps):
        x = model.attention @ x
        x = model.ffn @ x
    return steps / (time.time() - start)


def main() -> None:
    model = _DummyModel()
    adapter = HotSwapLoRAAdapter(model)
    # Create trivial weight deltas and apply them
    data = {
        "attention": (np.eye(model.attention.shape[0]) * 0.01).tolist(),
        "ffn": (np.eye(model.ffn.shape[0]) * 0.01).tolist(),
    }
    adapter.load_from_dict(data)

    # Benchmark baseline model
    baseline_model = _DummyModel()
    base_tps = _benchmark(baseline_model)
    adapted_tps = _benchmark(model)

    print(f"baseline: {base_tps:.2f} it/s")
    print(f"with adapter: {adapted_tps:.2f} it/s")

    # Save weights to file for demonstration
    with open("adapter_weights.json", "w", encoding="utf-8") as fh:
        json.dump(data, fh)


if __name__ == "__main__":
    main()
