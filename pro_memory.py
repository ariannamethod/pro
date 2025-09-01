"""
Упрощенная система памяти без MemoryStore.
Только SQLite + простой список сообщений.
"""

import asyncio
import numpy as np
from typing import List, Tuple, Optional, Set
from pro_memory_pool import init_pool, close_pool, get_connection

# Простая память на основе списка
_MESSAGES: List[Tuple[str, str]] = []

# Константы
DB_PATH = "pro_memory.db"


def _add_to_graph(content: str, msg_type: str, tag: str, embedding: Optional[np.ndarray] = None) -> None:
    """Добавляем в простой список сообщений."""
    _MESSAGES.append((content, tag))


async def init_db() -> None:
    """Initialize the database pool."""
    await init_pool(DB_PATH)
    
    # Создаем таблицы если их нет
    async with get_connection() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                tag TEXT DEFAULT 'message',
                fingerprint TEXT DEFAULT ''
            )
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS adapter_usage (
                name TEXT PRIMARY KEY,
                count INTEGER DEFAULT 0
            )
        """)
        
        await conn.commit()


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
