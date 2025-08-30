import asyncio
import atexit
from typing import Dict, List, Tuple, Optional

import numpy as np
import pro_rag_embedding
from pro_memory_pool import init_pool, close_pool, get_connection

DB_PATH = 'pro_memory.db'


_VECTORS: np.ndarray | None = None
_MESSAGES: List[str] = []
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


def _add_to_index(content: str, embedding: np.ndarray) -> None:
    """Add a vector to the in-memory search index."""
    global _VECTORS
    vec = embedding.reshape(1, -1)
    if _VECTORS is None:
        _VECTORS = vec
    else:
        _VECTORS = np.vstack([_VECTORS, vec])
    _MESSAGES.append(content)


async def build_index() -> None:
    """Load all stored embeddings into the in-memory index."""
    global _VECTORS, _MESSAGES
    async with get_connection() as conn:
        cur = await asyncio.to_thread(
            conn.execute, 'SELECT content, embedding FROM messages'
        )
        rows = await asyncio.to_thread(cur.fetchall)
    _MESSAGES = [r[0] for r in rows]
    if rows:
        _VECTORS = np.vstack(
            [np.frombuffer(r[1], dtype=np.float32) for r in rows]
        )
    else:
        _VECTORS = None


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


atexit.register(lambda: asyncio.run(close_db()))


async def add_message(content: str, tag: Optional[str] = None) -> None:
    """Encode a message, persist it, and update the search index."""
    embedding = await encode_message(content)
    await persist_embedding(content, embedding, tag)
    _add_to_index(content, embedding)


async def store_if_novel(
    content: str, threshold: float = 0.1, tag: Optional[str] = None
) -> bool:
    """Store ``content`` only if it is sufficiently different.

    The message is encoded and compared against existing embeddings using
    Euclidean distance.  If the closest stored message is within ``threshold``
    distance, the content is considered a duplicate and is not stored.

    Args:
        content: Text to potentially store.
        threshold: Minimum distance required to treat the message as novel.

    Returns:
        ``True`` if the message was stored, ``False`` otherwise.
    """
    if _VECTORS is None or not _MESSAGES:
        await build_index()

    embedding = await encode_message(content)
    if _VECTORS is not None:
        distances = np.linalg.norm(_VECTORS - embedding, axis=1)
        if distances.size and float(distances.min()) < threshold:
            return False

    await persist_embedding(content, embedding, tag)
    _add_to_index(content, embedding)
    return True


async def add_concept(description: str) -> None:
    """Extract entities and relations from text and store them."""
    entities, relations = await pro_rag_embedding.extract_entities_relations(
        description
    )
    if not entities:
        return
    async with get_connection() as conn:
        for ent in entities:
            await asyncio.to_thread(
                conn.execute,
                'INSERT OR IGNORE INTO concepts(name) VALUES (?)',
                (ent,),
            )
        for subj, rel, obj in relations:
            cur = await asyncio.to_thread(
                conn.execute, 'SELECT id FROM concepts WHERE name = ?', (subj,)
            )
            subj_id = await asyncio.to_thread(cur.fetchone)
            cur = await asyncio.to_thread(
                conn.execute, 'SELECT id FROM concepts WHERE name = ?', (obj,)
            )
            obj_id = await asyncio.to_thread(cur.fetchone)
            if subj_id and obj_id:
                await asyncio.to_thread(
                    conn.execute,
                    'INSERT INTO relations(source, target, relation) VALUES (?, ?, ?)',
                    (subj_id[0], obj_id[0], rel),
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
    if _VECTORS is None or not _MESSAGES:
        await build_index()
    if _VECTORS is None or not _MESSAGES:
        return []
    q_vec = await encode_message(query)
    vecs = _VECTORS
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    vecs_norm = vecs / norms
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-10)
    sims = vecs_norm @ q_norm
    top = np.argsort(sims)[-top_k:][::-1]
    return [_MESSAGES[i] for i in top]


async def fetch_related_concepts(words: List[str]) -> List[str]:
    """Return relation sentences touching any of the given words."""
    results: List[str] = []
    seen = set()
    terms = [w.lower() for w in words]
    async with get_connection() as conn:
        for term in terms:
            cur = await asyncio.to_thread(
                conn.execute,
                (
                    "SELECT c1.name, r.relation, c2.name FROM relations r "
                    "JOIN concepts c1 ON r.source = c1.id "
                    "JOIN concepts c2 ON r.target = c2.id "
                    "WHERE LOWER(c1.name) = ? OR LOWER(c2.name) = ?"
                ),
                (term, term),
            )
            rows = await asyncio.to_thread(cur.fetchall)
            for s, rel, t in rows:
                sentence = f"{s} {rel} {t}"
                if sentence not in seen:
                    seen.add(sentence)
                    results.append(sentence)
    return results
