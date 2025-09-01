"""Python 3.7 совместимость"""

import asyncio
import concurrent.futures

# Python 3.7 совместимость для asyncio.to_thread
async def to_thread(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))
