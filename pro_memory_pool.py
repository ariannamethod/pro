import asyncio
import atexit
import sqlite3
from contextlib import asynccontextmanager
from typing import List

_DB_PATH: str | None = None
_POOL: List[sqlite3.Connection] = []
_LOCK = asyncio.Lock()


async def init_pool(db_path: str, size: int = 1) -> None:
    """(Re)initialize connection pool and create tables if needed."""
    global _DB_PATH
    _DB_PATH = db_path
    async with _LOCK:
        while _POOL:
            _POOL.pop().close()
        for _ in range(size):
            conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
            _POOL.append(conn)
        conn = _POOL[0]
        conn.execute(
            "CREATE TABLE IF NOT EXISTS messages("  # noqa: E501
            "id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT, embedding BLOB)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS responses("  # noqa: E501
            "id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS concepts("  # noqa: E501
            "id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS relations("  # noqa: E501
            "id INTEGER PRIMARY KEY AUTOINCREMENT, source INTEGER, target INTEGER, relation TEXT,"
            "FOREIGN KEY(source) REFERENCES concepts(id),"
            "FOREIGN KEY(target) REFERENCES concepts(id))"
        )
        # Ensure embedding column exists for pre-existing databases
        try:
            conn.execute("ALTER TABLE messages ADD COLUMN embedding BLOB")
        except sqlite3.OperationalError:
            pass
        conn.commit()


@asynccontextmanager
async def get_connection():
    """Yield a connection from the pool."""
    if _DB_PATH is None:
        raise RuntimeError("Pool not initialized")
    async with _LOCK:
        if not _POOL:
            conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        else:
            conn = _POOL.pop()
    try:
        yield conn
    finally:
        async with _LOCK:
            _POOL.append(conn)


async def close_pool() -> None:
    """Close all pooled connections."""
    async with _LOCK:
        while _POOL:
            _POOL.pop().close()


atexit.register(lambda: asyncio.run(close_pool()))
