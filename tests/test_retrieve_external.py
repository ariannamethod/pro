import os
import sys
import asyncio

import pytest
from aiohttp import web

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pro_rag  # noqa: E402


@pytest.mark.asyncio
async def test_retrieve_external_success(monkeypatch):
    pro_rag._external_cache.clear()
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
    pro_rag._external_cache.clear()
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


@pytest.mark.asyncio
async def test_retrieve_external_cache(monkeypatch):
    pro_rag._external_cache.clear()
    app = web.Application()
    calls = {"count": 0}

    async def handler(request):
        calls["count"] += 1
        return web.json_response(["test", [], ["desc"], []])

    app.router.add_get("/", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    monkeypatch.setenv("WIKIPEDIA_API", f"http://localhost:{port}")
    try:
        result1 = await pro_rag.retrieve_external("cache")
        result2 = await pro_rag.retrieve_external("cache")
    finally:
        await runner.cleanup()
    assert result1 == ["desc"]
    assert result2 == ["desc"]
    assert calls["count"] == 1
