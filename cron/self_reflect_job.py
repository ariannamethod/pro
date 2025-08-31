"""Periodic background job for self-reflection fine-tuning.

The job limits GPU usage to avoid exhausting resources and then runs the
:class:`~self_reflect.SelfFineTuner` on a fixed interval.
"""

import os
import time
from typing import List, Optional

# Torch is optional; when unavailable we fall back to CPU-only behaviour.
try:  # pragma: no cover - best effort import
    import torch
except Exception:  # pragma: no cover - torch may not be installed
    torch = None  # type: ignore[assignment]

from self_reflect import SelfFineTuner

GPU_FRACTION = float(os.getenv("SELF_REFLECT_GPU_FRACTION", "0.5"))
INTERVAL_SECONDS = int(os.getenv("SELF_REFLECT_INTERVAL", "86400"))


def run_cycle(conversations: Optional[List[str]] = None) -> None:
    """Run a single self-reflection cycle with optional GPU limits."""
    if (
        torch is not None
        and hasattr(torch, "cuda")
        and torch.cuda.is_available()
    ):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        torch.cuda.set_per_process_memory_fraction(GPU_FRACTION, 0)
        tuner = SelfFineTuner()
    else:  # CPU-only path when torch or CUDA is unavailable
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        tuner = SelfFineTuner()
    tuner.run(conversations or [])


if __name__ == "__main__":
    while True:
        run_cycle([])
        time.sleep(INTERVAL_SECONDS)
