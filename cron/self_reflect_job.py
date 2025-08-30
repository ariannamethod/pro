"""Periodic background job for self-reflection fine-tuning.

The job limits GPU usage to avoid exhausting resources and then runs the
:class:`~self_reflect.SelfFineTuner` on a fixed interval.
"""

import os
import time
from typing import List, Optional

import torch

from self_reflect import SelfFineTuner

GPU_FRACTION = float(os.getenv("SELF_REFLECT_GPU_FRACTION", "0.5"))
INTERVAL_SECONDS = int(os.getenv("SELF_REFLECT_INTERVAL", "86400"))


def run_cycle(conversations: Optional[List[str]] = None) -> None:
    """Run a single self-reflection cycle with GPU limits."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(GPU_FRACTION, 0)
    tuner = SelfFineTuner()
    tuner.run(conversations or [])


if __name__ == "__main__":
    while True:
        run_cycle([])
        time.sleep(INTERVAL_SECONDS)
