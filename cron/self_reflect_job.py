"""Periodic background job for self-reflection fine-tuning.

Runs :class:`~self_reflect.SelfFineTuner` on a fixed interval.
"""

import asyncio
import os
from typing import List, Optional

from self_reflect import SelfFineTuner

INTERVAL_SECONDS = int(os.getenv("SELF_REFLECT_INTERVAL", "86400"))


async def run_cycle(conversations: Optional[List[str]] = None) -> None:
    """Run a single self-reflection cycle."""
    tuner = SelfFineTuner()
    tuner.run(conversations or [], {})


async def main() -> None:
    while True:
        await run_cycle([])
        await asyncio.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    asyncio.run(main())
