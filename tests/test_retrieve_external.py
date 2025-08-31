import os
import sys
import asyncio

import pytest
from aiohttp import web

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pro_rag  # noqa: E402


@pytest.mark.asyncio
async def test_retrieve_external_success(monkeypatch):
    app = web.Application()

    async def handler(request):
        return web.json_response(["test", [], ["desc"], []])

    app.router.add_get("/", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    monkeypatch.setenv("WIKIPEDIA_API", f"http://localhost:{port}")
    try:
        result = await pro_rag.retrieve_external("test")
    finally:
        await runner.cleanup()
    assert result == ["desc"]


@pytest.mark.asyncio
async def test_retrieve_external_timeout(monkeypatch):
    app = web.Application()

    async def handler(request):
        await asyncio.sleep(0.2)
        return web.json_response(["test", [], ["desc"], []])

    app.router.add_get("/", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    monkeypatch.setenv("WIKIPEDIA_API", f"http://localhost:{port}")
    monkeypatch.setenv("RAG_EXTERNAL_TIMEOUT", "0.05")
    try:
        result = await pro_rag.retrieve_external("test")
    finally:
        await runner.cleanup()
    assert result == []
