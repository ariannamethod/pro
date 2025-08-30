import asyncio
import json
import os
import hashlib
from typing import List
from urllib import request

DEFAULT_URL = os.getenv("VECTOR_STORE_URL")

async def upsert(text: str, embedding: List[float], base_url: str | None = None) -> None:
    """Store ``embedding`` for ``text`` in the external vector store.

    The vector store is expected to expose a JSON API with an ``/upsert``
    endpoint accepting ``{"id": str, "embedding": List[float], "text": str}``.
    The call is executed in a thread to avoid blocking the event loop.
    """
    base_url = base_url or DEFAULT_URL
    if not base_url:
        return

    url = f"{base_url}/upsert"
    payload = json.dumps({
        "id": hashlib.sha1(text.encode("utf-8")).hexdigest(),
        "embedding": embedding,
        "text": text,
    }).encode("utf-8")

    def _request() -> None:
        req = request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with request.urlopen(req) as resp:  # pragma: no cover - network side effect
            resp.read()

    await asyncio.to_thread(_request)

async def query(embedding: List[float], top_k: int = 5, base_url: str | None = None) -> List[str]:
    """Return texts most similar to ``embedding`` from the external store."""
    base_url = base_url or DEFAULT_URL
    if not base_url:
        return []

    url = f"{base_url}/query"
    payload = json.dumps({"embedding": embedding, "top_k": top_k}).encode("utf-8")

    def _request() -> List[str]:
        req = request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with request.urlopen(req) as resp:  # pragma: no cover - network side effect
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("texts", [])

    return await asyncio.to_thread(_request)
