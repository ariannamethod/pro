"""Unified memory storage and retrieval utilities."""

from __future__ import annotations

import asyncio
import atexit
import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from autograd import numpy as anp

import pro_rag_embedding
from morphology import encode as encode_morph

from .pooling import init_pool, close_pool, get_connection


@dataclass
class MemoryNode:
    """Single utterance optionally backed by an embedding vector."""

    speaker: str
    text: str
    index: Optional[int] = None
    embedding: Optional[Dict[str, float]] = None
    morph_codes: Optional[List[float]] = None


class MemoryStore:
    """Store dialogues with optional embeddings for retrieval."""

    def __init__(self, dim: int = 32, path: Optional[str] = None) -> None:
        self.dim = dim
        self.path = path
        self.graph: Dict[str, List[MemoryNode]] = {}
        self.embeddings = anp.zeros((0, dim))
        if path and os.path.exists(path):
            self.load(path)

    # ------------------------------------------------------------------ storage
    def add_utterance(
        self,
        dialogue_id: str,
        speaker: str,
        text: str,
        embedding: Optional[Any] = None,
    ) -> Optional[anp.ndarray]:
        """Append an utterance to a dialogue."""

        if isinstance(embedding, dict):
            node = MemoryNode(speaker, text, embedding=embedding)
            self.graph.setdefault(dialogue_id, []).append(node)
            if self.path:
                self.save(self.path)
            return None

        vec = anp.array(embedding if embedding is not None else encode_morph(text, self.dim))
        if self.embeddings.size:
            self.embeddings = anp.vstack([self.embeddings, vec])
        else:
            self.embeddings = vec.reshape(1, -1)
        idx = self.embeddings.shape[0] - 1
        node = MemoryNode(speaker, text, index=idx, morph_codes=vec.tolist())
        self.graph.setdefault(dialogue_id, []).append(node)
        return self.embeddings[idx]

    def get_dialogue(self, dialogue_id: str) -> List[MemoryNode]:
        return list(self.graph.get(dialogue_id, []))

    # ---------------------------------------------------------------- embeddings
    def embeddings_for(self, dialogue_id: str, speaker: Optional[str] = None) -> anp.ndarray:
        idxs = [
            n.index
            for n in self.graph.get(dialogue_id, [])
            if n.index is not None and (speaker is None or n.speaker == speaker)
        ]
        if not idxs:
            return anp.zeros((0, self.dim))
        return self.embeddings[idxs]

    def retrieve(
        self,
        query: anp.ndarray,
        dialogue_id: str,
        speaker: Optional[str] = None,
        embeddings: Optional[anp.ndarray] = None,
    ) -> anp.ndarray:
        """Return a weighted memory vector for ``query`` using softmax weighting."""

        vecs = embeddings if embeddings is not None else self.embeddings_for(dialogue_id, speaker)
        if vecs.shape[0] == 0:
            return anp.zeros(self.dim)
        scores = vecs @ query
        scores = scores - anp.max(scores)
        weights = anp.exp(scores)
        weights = weights / anp.sum(weights)
        return weights @ vecs

    # ---------------------------------------------------------------- persistence
    def save(self, path: Optional[str] = None) -> None:
        path = path or self.path
        if not path:
            return
        data = {
            did: [asdict(node) for node in nodes]
            for did, nodes in self.graph.items()
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        self.path = path

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        self.graph = {}
        vecs: List[List[float]] = []
        for did, nodes in raw.items():
            restored: List[MemoryNode] = []
            for node in nodes:
                mn = MemoryNode(**node)
                if mn.morph_codes is not None:
                    mn.index = len(vecs)
                    vecs.append(mn.morph_codes)
                restored.append(mn)
            self.graph[did] = restored
        self.embeddings = anp.array(vecs) if vecs else anp.zeros((0, self.dim))
        self.path = path


class GraphRetriever:
    """Helper that provides high level queries over :class:`MemoryStore`."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def last_message(self, dialogue_id: str, speaker: str) -> Optional[str]:
        for node in reversed(self.store.get_dialogue(dialogue_id)):
            if node.speaker == speaker:
                return node.text
        return None

    def all_messages(self, dialogue_id: str) -> List[str]:
        return [n.text for n in self.store.get_dialogue(dialogue_id)]

    def retrieve(
        self, dialogue_id: str, speaker: str, query: Optional[anp.ndarray] = None
    ) -> anp.ndarray:
        if query is None:
            query = anp.zeros(self.store.dim)
        return self.store.retrieve(query, dialogue_id, speaker)


# ---------------------------------------------------------------------- high level
DB_PATH = "pro_memory.db"
_STORE: MemoryStore | None = None
_ADAPTER_USAGE: Dict[str, int] = {}


async def init_db() -> None:
    """Initialize database and connection pool."""
    await init_pool(DB_PATH)


async def encode_message(content: str) -> np.ndarray:
    """Encode text into an embedding vector using a transformer model."""
    embedding = await pro_rag_embedding.embed_sentence(content)
    return embedding.astype(np.float32)


async def persist_embedding(
    content: str, embedding: np.ndarray, tag: Optional[str] = None
) -> None:
    """Persist a message and its embedding to the database."""
    async with get_connection() as conn:
        await asyncio.to_thread(
            conn.execute,
            'INSERT INTO messages(content, embedding, tag) VALUES (?, ?, ?)',
            (content, embedding.tobytes(), tag),
        )
        await asyncio.to_thread(conn.commit)


async def persist_learned_vector(index: int, embedding: np.ndarray) -> None:
    """Persist an updated embedding for an existing message."""
    async with get_connection() as conn:
        await asyncio.to_thread(
            conn.execute,
            "UPDATE messages SET embedding = ? WHERE rowid = ?",
            (embedding.tobytes(), index + 1),
        )
        await asyncio.to_thread(conn.commit)
    if _STORE is not None and index < _STORE.embeddings.shape[0]:
        _STORE.embeddings[index] = embedding


def _add_to_index(content: str, embedding: np.ndarray) -> None:
    """Add a vector to the in-memory search index."""
    global _STORE
    if _STORE is None:
        _STORE = MemoryStore(dim=embedding.shape[0])
    _STORE.add_utterance("global", "user", content, embedding)


async def build_index() -> None:
    """Load all stored embeddings into the in-memory index."""
    global _STORE
    async with get_connection() as conn:
        cur = await asyncio.to_thread(
            conn.execute, 'SELECT content, embedding FROM messages'
        )
        rows = await asyncio.to_thread(cur.fetchall)
    if not rows:
        _STORE = None
        return
    first_vec = np.frombuffer(rows[0][1], dtype=np.float32)
    _STORE = MemoryStore(dim=first_vec.shape[0])
    for content, blob in rows:
        vec = np.frombuffer(blob, dtype=np.float32)
        _STORE.add_utterance("global", "user", content, vec)


async def close_db() -> None:
    """Close all connections in the pool."""
    await close_pool()


async def increment_adapter_usage(name: str) -> None:
    """Increment usage counter for a specific adapter."""
    _ADAPTER_USAGE[name] = _ADAPTER_USAGE.get(name, 0) + 1
    async with get_connection() as conn:
        await asyncio.to_thread(
            conn.execute,
            "INSERT INTO adapter_usage(adapter, count) VALUES (?, 1) "
            "ON CONFLICT(adapter) DO UPDATE SET count = count + 1",
            (name,),
        )
        await asyncio.to_thread(conn.commit)


async def total_adapter_usage() -> int:
    """Return the total number of adapter usages recorded."""
    async with get_connection() as conn:
        cur = await asyncio.to_thread(
            conn.execute, "SELECT SUM(count) FROM adapter_usage"
        )
        row = await asyncio.to_thread(cur.fetchone)
    return int(row[0] or 0)


async def reset_adapter_usage() -> None:
    """Clear all adapter usage counters."""
    _ADAPTER_USAGE.clear()
    async with get_connection() as conn:
        await asyncio.to_thread(conn.execute, "DELETE FROM adapter_usage")
        await asyncio.to_thread(conn.commit)


def _close_db_sync() -> None:
    """Synchronously close the database connection pool."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(close_db())
    else:
        loop.create_task(close_db())


atexit.register(_close_db_sync)


async def add_message(content: str, tag: Optional[str] = None) -> None:
    """Encode a message, persist it, and update the search index."""
    embedding = await encode_message(content)
    await persist_embedding(content, embedding, tag)
    _add_to_index(content, embedding)


async def store_if_novel(
    content: str, threshold: float = 0.1, tag: Optional[str] = None
) -> bool:
    """Store ``content`` only if it is sufficiently different."""
    if _STORE is None or _STORE.embeddings.shape[0] == 0:
        await build_index()
    embedding = await encode_message(content)
    if _STORE is not None and _STORE.embeddings.shape[0]:
        distances = np.linalg.norm(_STORE.embeddings - embedding, axis=1)
        if distances.size and float(distances.min()) < threshold:
            return False
    await persist_embedding(content, embedding, tag)
    _add_to_index(content, embedding)
    return True


async def is_unique(sentence: str) -> bool:
    """Return True if sentence not already stored."""
    async with get_connection() as conn:
        cur = await asyncio.to_thread(
            conn.execute,
            'SELECT 1 FROM responses WHERE content = ? LIMIT 1',
            (sentence,),
        )
        exists = await asyncio.to_thread(cur.fetchone)
    return exists is None


async def store_response(sentence: str, tag: Optional[str] = None) -> None:
    """Persist a generated response."""
    async with get_connection() as conn:
        await asyncio.to_thread(
            conn.execute,
            'INSERT INTO responses(content, tag) VALUES (?, ?)',
            (sentence, tag),
        )
        await asyncio.to_thread(conn.commit)


async def fetch_recent(limit: int = 5) -> Tuple[List[str], List[str]]:
    """Fetch recent messages and responses for context."""
    async with get_connection() as conn:
        cur = await asyncio.to_thread(
            conn.execute,
            'SELECT content FROM messages ORDER BY id DESC LIMIT ?',
            (limit,),
        )
        msg_rows = await asyncio.to_thread(cur.fetchall)
        cur = await asyncio.to_thread(
            conn.execute,
            'SELECT content FROM responses ORDER BY id DESC LIMIT ?',
            (limit,),
        )
        resp_rows = await asyncio.to_thread(cur.fetchall)
    messages = [r[0] for r in msg_rows][::-1]
    responses = [r[0] for r in resp_rows][::-1]
    return messages, responses


async def fetch_recent_messages(limit: int = 50) -> List[Tuple[str, np.ndarray]]:
    """Fetch recent messages with embeddings."""
    async with get_connection() as conn:
        cur = await asyncio.to_thread(
            conn.execute,
            'SELECT content, embedding FROM messages ORDER BY id DESC LIMIT ?',
            (limit,),
        )
        rows = await asyncio.to_thread(cur.fetchall)
    messages = [
        (r[0], np.frombuffer(r[1], dtype=np.float32)) for r in rows
    ][::-1]
    return messages


async def fetch_similar_messages(query: str, top_k: int = 5) -> List[str]:
    """Return top-k stored messages most similar to the query."""
    if _STORE is None or _STORE.embeddings.shape[0] == 0:
        await build_index()
    if _STORE is None or _STORE.embeddings.shape[0] == 0:
        return []
    q_vec = await encode_message(query)
    vecs = _STORE.embeddings
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    vecs_norm = vecs / norms
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-10)
    sims = vecs_norm @ q_norm
    top = np.argsort(sims)[-top_k:][::-1]
    nodes = _STORE.graph.get("global", [])
    return [nodes[i].text for i in top]
