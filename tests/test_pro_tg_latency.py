import os
import sys
import time
from pathlib import Path

import aiohttp
from aiohttp import web
import pytest

# Ensure token for module import
os.environ.setdefault("TELEGRAM_TOKEN", "TOKEN")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pro_tg  # noqa: E402


@pytest.mark.asyncio
async def test_telegram_latency_under_5s():
    app = web.Application()

    async def handle_get_updates(request):
        return web.json_response({"ok": True, "result": []})

    async def handle_send_message(request):
        data = await request.json()
        return web.json_response({"ok": True, "result": data})

    app.router.add_get("/botTOKEN/getUpdates", handle_get_updates)
    app.router.add_post("/botTOKEN/sendMessage", handle_send_message)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    pro_tg.API_URL = f"http://localhost:{port}/botTOKEN"

    async with aiohttp.ClientSession() as session:
        start = time.perf_counter()
        await pro_tg.get_updates(session)
        await pro_tg.send_message(session, 1, "hi")
        elapsed = time.perf_counter() - start

    await runner.cleanup()

    assert elapsed < 5
