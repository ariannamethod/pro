import asyncio
import atexit
import sqlite3
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Tuple

_DB_PATH: str | None = None
_POOL: List[sqlite3.Connection] = []
_LOCK = asyncio.Lock()
_CACHE: Dict[Tuple[str, Tuple[Any, ...]], Tuple[float, Any]] = {}


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
            "id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT, embedding BLOB, tag TEXT)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS responses("  # noqa: E501
            "id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT, tag TEXT)"
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
        conn.execute(
            "CREATE TABLE IF NOT EXISTS adapter_usage("  # noqa: E501
            "adapter TEXT PRIMARY KEY, count INTEGER)"
        )
        # Ensure new columns exist for pre-existing databases
        for stmt in [
            "ALTER TABLE messages ADD COLUMN embedding BLOB",
            "ALTER TABLE messages ADD COLUMN tag TEXT",
            "ALTER TABLE responses ADD COLUMN tag TEXT",
        ]:
            try:
                conn.execute(stmt)
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
    clear_cache()


def clear_cache() -> None:
    """Remove all cached query results."""
    _CACHE.clear()


async def execute_cached(
    query: str, params: Tuple[Any, ...] | List[Any] | None = None, ttl: float = 30.0
) -> List[Any]:
    """Execute a read-only ``query`` with TTL-based caching.

    Args:
        query: SQL query to execute.
        params: Parameters for the query.
        ttl: Number of seconds the result should remain cached.

    Returns:
        List of rows returned by the query.
    """
    if params is None:
        params_tuple: Tuple[Any, ...] = ()
    elif isinstance(params, tuple):
        params_tuple = params
    else:
        params_tuple = tuple(params)
    key = (query, params_tuple)
    now = time.monotonic()
    cached = _CACHE.get(key)
    if cached and cached[0] > now:
        return cached[1]
    async with get_connection() as conn:
        cur = await asyncio.to_thread(conn.execute, query, params_tuple)
        rows = await asyncio.to_thread(cur.fetchall)
    _CACHE[key] = (now + ttl, rows)
    return rows


atexit.register(lambda: asyncio.run(close_pool()))
