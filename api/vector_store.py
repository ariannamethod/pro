import os
import hashlib
from typing import List

import aiohttp

DEFAULT_URL = os.getenv("VECTOR_STORE_URL")

async def upsert(text: str, embedding: List[float], base_url: str | None = None) -> None:
    """Store ``embedding`` for ``text`` in the external vector store.

    The vector store is expected to expose a JSON API with an ``/upsert``
    endpoint accepting ``{"id": str, "embedding": List[float], "text": str}``.
    """
    base_url = base_url or DEFAULT_URL
    if not base_url:
        return

    url = f"{base_url}/upsert"
    payload = {
        "id": hashlib.sha1(text.encode("utf-8")).hexdigest(),
        "embedding": embedding,
        "text": text,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:  # pragma: no cover - network side effect
            await resp.read()

async def query(embedding: List[float], top_k: int = 5, base_url: str | None = None) -> List[str]:
    """Return texts most similar to ``embedding`` from the external store."""
    base_url = base_url or DEFAULT_URL
    if not base_url:
        return []

    url = f"{base_url}/query"
    payload = {"embedding": embedding, "top_k": top_k}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:  # pragma: no cover - network side effect
            data = await resp.json()
    return data.get("texts", [])
