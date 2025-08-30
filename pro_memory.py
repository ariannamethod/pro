import asyncio
import atexit
from typing import List, Tuple

import numpy as np
import pro_rag_embedding
from pro_memory_pool import init_pool, close_pool, get_connection

DB_PATH = 'pro_memory.db'


async def init_db() -> None:
    """Initialize database and connection pool."""
    await init_pool(DB_PATH)


async def close_db() -> None:
    """Close all connections in the pool."""
    await close_pool()


atexit.register(lambda: asyncio.run(close_db()))


async def add_message(content: str) -> None:
    """Store a message and its embedding asynchronously."""
    embedding = await pro_rag_embedding.embed_sentence(content)
    async with get_connection() as conn:
        await asyncio.to_thread(
            conn.execute,
            'INSERT INTO messages(content, embedding) VALUES (?, ?)',
            (content, embedding.tobytes()),
        )
        await asyncio.to_thread(conn.commit)


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


async def store_response(sentence: str) -> None:
    """Persist a generated response."""
    async with get_connection() as conn:
        await asyncio.to_thread(
            conn.execute, 'INSERT INTO responses(content) VALUES (?)', (sentence,)
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
