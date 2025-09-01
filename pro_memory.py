"""
Упрощенная система памяти без MemoryStore.
Только SQLite + простой список сообщений.
Объединен с pro_memory_pool.
"""

import asyncio
import atexit
import aiosqlite
import time
import numpy as np
from contextlib import asynccontextmanager
from typing import List, Tuple, Optional, Set, Dict, Any, Union

# Простая память на основе списка
_MESSAGES: List[Tuple[str, str]] = []

# Константы
DB_PATH = "pro_memory.db"

# Пул соединений (из pro_memory_pool)
_DB_PATH: Optional[str] = None
_POOL: List[aiosqlite.Connection] = []
_LOCK = asyncio.Lock()
_CACHE: Dict[Tuple[str, Tuple[Any, ...]], Tuple[float, Any]] = {}


def _add_to_graph(content: str, msg_type: str, tag: str, embedding: Optional[np.ndarray] = None) -> None:
    """Добавляем в простой список сообщений."""
    _MESSAGES.append((content, tag))


async def init_pool(db_path: str, size: int = 1) -> None:
    """(Re)initialize connection pool and create tables if needed."""
    global _DB_PATH
    _DB_PATH = db_path
    async with _LOCK:
        while _POOL:
            conn = _POOL.pop()
            await conn.close()
        for _ in range(size):
            conn = await aiosqlite.connect(_DB_PATH)
            _POOL.append(conn)
        conn = _POOL[0]
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS messages("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT, embedding BLOB, tag TEXT, fingerprint TEXT)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_fingerprint ON messages(fingerprint)"
        )
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS responses("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT, tag TEXT)"
        )
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS concepts("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE)"
        )
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS relations("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, source INTEGER, target INTEGER, relation TEXT,"
            "FOREIGN KEY(source) REFERENCES concepts(id),"
            "FOREIGN KEY(target) REFERENCES concepts(id))"
        )
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS adapter_usage("
            "adapter TEXT PRIMARY KEY, count INTEGER)"
        )
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                tag TEXT DEFAULT 'message',
                fingerprint TEXT DEFAULT ''
            )
        """)
        await conn.commit()


@asynccontextmanager
async def get_connection():
    """Yield a connection from the pool."""
    if _DB_PATH is None:
        raise RuntimeError("Pool not initialized")
    async with _LOCK:
        if not _POOL:
            conn = await aiosqlite.connect(_DB_PATH)
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
            conn = _POOL.pop()
            await conn.close()
    clear_cache()


def clear_cache() -> None:
    """Remove all cached query results."""
    _CACHE.clear()


async def execute_cached(
    query: str, params: Optional[Union[Tuple[Any, ...], List[Any]]] = None, ttl: float = 30.0
) -> List[Any]:
    """Execute a read-only query with TTL-based caching."""
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
        async with conn.execute(query, params_tuple) as cur:
            rows = await cur.fetchall()
    _CACHE[key] = (now + ttl, rows)
    return rows


def _close_pool_sync() -> None:
    """Synchronously close the connection pool."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(close_pool())
    else:
        loop.create_task(close_pool())


atexit.register(_close_pool_sync)


async def init_db() -> None:
    """Initialize the database pool."""
    await init_pool(DB_PATH)


async def close_db() -> None:
    """Close all connections in the pool."""
    await close_pool()


async def encode_message(content: str) -> np.ndarray:
    """Encode message to vector using simple TF-IDF-like approach."""
    words = content.lower().split()
    # Простое кодирование: частоты слов
    vocab = list(set(words))
    if not vocab:
        return np.zeros(64, dtype=np.float32)
    
    # TF-IDF подобное кодирование
    vec = np.zeros(64, dtype=np.float32)
    for i, word in enumerate(vocab[:64]):
        freq = words.count(word) / len(words)
        vec[i] = freq
    
    # Нормализация
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    
    # Преобразование комплексных в вещественные
    if np.iscomplexobj(vec):
        vec = vec.real
    
    return vec.astype(np.float32)


async def build_index() -> None:
    """Построить индекс из SQLite."""
    # Простая версия - только SQLite, без MemoryStore
    pass


async def persist_embedding(content: str, embedding: np.ndarray, tag: str = "message", fingerprint: str = "") -> None:
    """Persist embedding to database."""
    async with get_connection() as conn:
        await conn.execute(
            "INSERT OR REPLACE INTO embeddings (content, embedding, tag, fingerprint) VALUES (?, ?, ?, ?)",
            (content, embedding.tobytes(), tag, fingerprint)
        )
        await conn.commit()


def _add_to_index(content: str, embedding: np.ndarray) -> None:
    """Add to simple index."""
    # Простая версия - ничего не делаем
    pass


async def is_unique_message(content: str, threshold: float = 0.85, tag: str = "message", fingerprint: str = "") -> bool:
    """Check if message is unique (always True in simplified version)."""
    # Упрощенная версия - всегда уникально
    await build_index()
    embedding = await encode_message(content)
    
    # Сохраняем
    await persist_embedding(content, embedding, tag, fingerprint)
    _add_to_index(content, embedding)
    _add_to_graph(content, "message", tag, embedding)
    return True


async def fetch_recent(limit: int = 5) -> List[str]:
    """Fetch recent messages from simple list."""
    return [msg[0] for msg in _MESSAGES[-limit:]]


async def fetch_recent_messages(limit: int = 10) -> List[str]:
    """Fetch recent messages."""
    async with get_connection() as conn:
        cursor = await conn.execute(
            "SELECT content FROM embeddings WHERE tag = 'message' ORDER BY rowid DESC LIMIT ?",
            (limit,)
        )
        rows = await cursor.fetchall()
        return [row[0] for row in rows]


async def fetch_recent_responses(limit: int = 10) -> List[str]:
    """Fetch recent responses."""
    async with get_connection() as conn:
        cursor = await conn.execute(
            "SELECT content FROM embeddings WHERE tag = 'response' ORDER BY rowid DESC LIMIT ?",
            (limit,)
        )
        rows = await cursor.fetchall()
        return [row[0] for row in rows]


async def fetch_similar_messages(query: str, top_k: int = 5, threshold: float = 0.7) -> List[str]:
    """Упрощенная версия - возвращает пустой список."""
    return []


async def fetch_related_concepts(words: List[str]) -> List[str]:
    """Return relation sentences touching any of the given words."""
    results: List[str] = []
    if not words:
        return results
    
    # Ищем в простом списке
    for content, tag in _MESSAGES:
        for word in words:
            if word.lower() in content.lower():
                results.append(content)
                break
    
    return results[:10]


async def increment_adapter_usage(name: str) -> None:
    """Increment adapter usage count."""
    async with get_connection() as conn:
        await conn.execute(
            "INSERT OR REPLACE INTO adapter_usage (name, count) VALUES (?, COALESCE((SELECT count FROM adapter_usage WHERE name = ?), 0) + 1)",
            (name, name)
        )
        await conn.commit()


async def get_adapter_stats() -> dict:
    """Get adapter usage statistics."""
    async with get_connection() as conn:
        cursor = await conn.execute("SELECT name, count FROM adapter_usage ORDER BY count DESC")
        rows = await cursor.fetchall()
        return {name: count for name, count in rows}


def get_memory_stats() -> dict:
    """Get memory statistics."""
    return {
        "total_messages": len(_MESSAGES),
        "recent_count": min(10, len(_MESSAGES))
    }


async def add_message(content: str, tag: str = "message") -> None:
    """Add message to memory."""
    embedding = await encode_message(content)
    await persist_embedding(content, embedding, tag)
    _add_to_graph(content, "message", tag, embedding)


async def is_unique(content: str, threshold: float = 0.85) -> bool:
    """Check if content is unique (simplified version - always True)."""
    return True


async def store_response(content: str) -> None:
    """Store response to memory."""
    await add_message(content, "response")
