import asyncio
import atexit
import sqlite3
from typing import List, Tuple

import aiosqlite

DB_PATH = 'pro_memory.db'
_CONN: aiosqlite.Connection | None = None


async def init_db() -> None:
    """Initialize sqlite database and create a global connection."""

    global _CONN
    _CONN = await aiosqlite.connect(DB_PATH)
    await _CONN.execute(
        "CREATE TABLE IF NOT EXISTS messages("  # noqa: E501
        "id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT)"
    )
    await _CONN.execute(
        "CREATE TABLE IF NOT EXISTS responses("  # noqa: E501
        "id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT)"
    )
    await _CONN.commit()


async def close_db() -> None:
    """Close the global database connection."""

    global _CONN
    if _CONN is not None:
        await _CONN.close()
        _CONN = None


atexit.register(lambda: asyncio.run(close_db()))


async def add_message(content: str) -> None:
    """Store a message asynchronously."""

    if _CONN is None:
        raise RuntimeError('Database not initialized')
    await _CONN.execute('INSERT INTO messages(content) VALUES (?)', (content,))
    await _CONN.commit()


def is_unique(sentence: str) -> bool:
    """Return True if sentence not already stored."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        'SELECT 1 FROM responses WHERE content = ? LIMIT 1',
        (sentence,),
    )
    exists = cur.fetchone()
    conn.close()
    return exists is None


async def _store_response(sentence: str) -> None:
    if _CONN is None:
        raise RuntimeError('Database not initialized')
    await _CONN.execute('INSERT INTO responses(content) VALUES (?)', (sentence,))
    await _CONN.commit()


def store_response(sentence: str) -> None:
    """Persist a generated response."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_store_response(sentence))
    else:
        loop.create_task(_store_response(sentence))


async def fetch_recent(limit: int = 5) -> Tuple[List[str], List[str]]:
    """Fetch recent messages and responses for context."""

    if _CONN is None:
        raise RuntimeError('Database not initialized')

    async with _CONN.execute(
        'SELECT content FROM messages ORDER BY id DESC LIMIT ?', (limit,)
    ) as cur:
        msg_rows = await cur.fetchall()
    async with _CONN.execute(
        'SELECT content FROM responses ORDER BY id DESC LIMIT ?', (limit,)
    ) as cur:
        resp_rows = await cur.fetchall()
    messages = [r[0] for r in msg_rows][::-1]
    responses = [r[0] for r in resp_rows][::-1]
    return messages, responses
