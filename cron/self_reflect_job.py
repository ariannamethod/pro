"""Periodic background job for self-reflection fine-tuning.

Runs :class:`~self_reflect.SelfFineTuner` on a fixed interval.
"""

import asyncio
import os
from typing import List, Optional

# Torch is optional; when unavailable we fall back to CPU-only behaviour.
try:  # pragma: no cover - best effort import
    import torch
except Exception:  # pragma: no cover - torch may not be installed
    torch = None  # type: ignore[assignment]

from self_reflect import SelfFineTuner

INTERVAL_SECONDS = int(os.getenv("SELF_REFLECT_INTERVAL", "86400"))

active_handle: Optional[asyncio.TimerHandle] = None


async def run_cycle(conversations: Optional[List[str]] = None) -> None:
    """Run a single self-reflection cycle."""
    tuner = (
        SelfFineTuner(model=torch.device("cpu"))
        if torch is not None
        else SelfFineTuner()
    )
    tuner.run(conversations or [], {})


def schedule_next(
    conversations: Optional[List[str]] = None,
    *,
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> asyncio.TimerHandle:
    """Schedule the next self-reflection cycle."""

    loop = loop or asyncio.get_running_loop()

    async def _cycle() -> None:
        await run_cycle(conversations)
        schedule_next(conversations, loop=loop)

    global active_handle
    active_handle = loop.call_later(
        INTERVAL_SECONDS, lambda: asyncio.create_task(_cycle())
    )
    return active_handle


async def main() -> None:
    schedule_next([])
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
