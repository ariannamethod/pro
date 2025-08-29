import sqlite3
import asyncio
from typing import List

DB_PATH = 'pro_memory.db'

async def init_db() -> None:
    """Initialize sqlite database."""
    def _setup():
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute('CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT)')
        conn.commit()
        conn.close()
    await asyncio.to_thread(_setup)

async def add_message(content: str) -> None:
    """Store a message asynchronously."""
    def _write():
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute('INSERT INTO messages(content) VALUES (?)', (content,))
        conn.commit()
        conn.close()
    await asyncio.to_thread(_write)

async def fetch_recent(limit: int = 5) -> List[str]:
    """Fetch recent messages for context."""
    def _read() -> List[str]:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute('SELECT content FROM messages ORDER BY id DESC LIMIT ?', (limit,))
        rows = cur.fetchall()
        conn.close()
        return [r[0] for r in rows][::-1]
    return await asyncio.to_thread(_read)
