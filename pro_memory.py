import asyncio
import atexit
from typing import Dict, List, Tuple, Optional, Set

import hashlib
import numpy as np
import re
import morphology
import pro_rag_embedding
from pro_memory_pool import init_pool, close_pool, get_connection
from memory.store import MemoryStore
# HyperGraph удален - используем простую память

DB_PATH = 'pro_memory.db'

_MIN_TOKENS = 2
_BLACKLIST_PATTERNS = [re.compile(r'^v\d+\.\d+$', re.IGNORECASE)]


_STORE: Optional[MemoryStore] = None
_ADAPTER_USAGE: Dict[str, int] = {}
COMPRESSION_INTERVAL = 100
COMPRESSION_EVENT = asyncio.Event()
_TOTAL_ADAPTER_USAGE = 0

_MESSAGES: List[Tuple[str, str]] = []  # Простая память: (content, kind)


def _fingerprint(content: str) -> str:
    """Return a stable hash of the morphological tokens in ``content``."""
    tokens = morphology.tokenize(content)
    joined = " ".join(tokens)
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


def _is_short(content: str) -> bool:
    return len(re.findall(r"\w+", content)) < _MIN_TOKENS


def _is_blacklisted(content: str) -> bool:
    return any(p.fullmatch(content) for p in _BLACKLIST_PATTERNS)


async def init_db() -> None:
    """Initialize database and connection pool."""
    await init_pool(DB_PATH)


async def encode_message(content: str) -> np.ndarray:
    """Encode text into an embedding vector using a transformer model."""
    embedding = await pro_rag_embedding.embed_sentence(content)
    return embedding.astype(np.float32)


async def persist_embedding(
    content: str,
    embedding: np.ndarray,
    tag: Optional[str] = None,
    fingerprint: Optional[str] = None,
) -> None:
    """Persist a message, its embedding and fingerprint to the database."""
    async with get_connection() as conn:
        await conn.execute(
            'INSERT INTO messages(content, embedding, tag, fingerprint) VALUES (?, ?, ?, ?)',
            (content, embedding.tobytes(), tag, fingerprint),
        )
        await conn.commit()


async def persist_learned_vector(index: int, embedding: np.ndarray) -> None:
    """Persist an updated embedding for an existing message."""

    async with get_connection() as conn:
        await conn.execute(
            "UPDATE messages SET embedding = ? WHERE rowid = ?",
            (embedding.tobytes(), index + 1),
        )
        await conn.commit()
    if _STORE is not None and index < _STORE.embeddings.shape[0]:
        _STORE.embeddings[index] = embedding


def _add_to_index(content: str, embedding: np.ndarray) -> None:
    """Add a vector to the in-memory search index."""
    global _STORE
    if _STORE is None:
        _STORE = MemoryStore(dim=embedding.shape[0])
    _STORE.add_utterance("global", "user", content, embedding)


def _add_to_graph(
    content: str,
    kind: str,
    tag: Optional[str] = None,
    embedding: Optional[np.ndarray] = None,
) -> None:
    """Insert ``content`` into the conversation hypergraph."""

    _MESSAGES.append((content, kind))


async def build_index(batch_size: int = 100) -> None:
    """Load stored embeddings into the in-memory index in batches.

    Args:
        batch_size: Number of rows fetched per database query.
    """
    global _STORE
    offset = 0
    first_batch = True
    async with get_connection() as conn:
        while True:
            async with conn.execute(
                "SELECT content, embedding FROM messages LIMIT ? OFFSET ?",
                (batch_size, offset),
            ) as cur:
                rows = await cur.fetchall()
            if not rows:
                break
            if first_batch:
                first_vec = np.frombuffer(rows[0][1], dtype=np.float32)
                _STORE = MemoryStore(dim=first_vec.shape[0])
                first_batch = False
            for content, blob in rows:
                vec = np.frombuffer(blob, dtype=np.float32)
                _STORE.add_utterance("global", "user", content, vec)
            offset += batch_size
    if first_batch:
        _STORE = None


async def close_db() -> None:
    """Close all connections in the pool."""
    await close_pool()


async def increment_adapter_usage(name: str) -> None:
    """Increment usage counter for a specific adapter."""
    global _TOTAL_ADAPTER_USAGE
    _ADAPTER_USAGE[name] = _ADAPTER_USAGE.get(name, 0) + 1
    _TOTAL_ADAPTER_USAGE += 1
    async with get_connection() as conn:
        await conn.execute(
            "INSERT INTO adapter_usage(adapter, count) VALUES (?, 1) "
            "ON CONFLICT(adapter) DO UPDATE SET count = count + 1",
            (name,),
        )
        await conn.commit()
    if _TOTAL_ADAPTER_USAGE >= COMPRESSION_INTERVAL:
        COMPRESSION_EVENT.set()


async def total_adapter_usage() -> int:
    """Return the total number of adapter usages recorded."""
    async with get_connection() as conn:
        async with conn.execute(
            "SELECT SUM(count) FROM adapter_usage"
        ) as cur:
            row = await cur.fetchone()
    return int(row[0] or 0)


async def reset_adapter_usage() -> None:
    """Clear all adapter usage counters."""
    global _TOTAL_ADAPTER_USAGE
    _ADAPTER_USAGE.clear()
    _TOTAL_ADAPTER_USAGE = 0
    COMPRESSION_EVENT.clear()
    async with get_connection() as conn:
        await conn.execute("DELETE FROM adapter_usage")
        await conn.commit()


def _close_db_sync() -> None:
    """Synchronously close the database connection pool.

    If an event loop is already running, ``close_db`` is scheduled on that
    loop.  Otherwise, the coroutine is executed in a new event loop via
    :func:`asyncio.run`.
    """
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
    fingerprint = _fingerprint(content)
    await persist_embedding(content, embedding, tag, fingerprint)
    _add_to_index(content, embedding)
    _add_to_graph(content, "message", tag, embedding)


async def store_if_novel(
    content: str, threshold: float = 0.3, tag: Optional[str] = None
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
    content = content.strip()
    if _is_short(content) or _is_blacklisted(content):
        return False
    fingerprint = _fingerprint(content)
    async with get_connection() as conn:
        async with conn.execute(
            'SELECT 1 FROM messages WHERE fingerprint = ? LIMIT 1',
            (fingerprint,),
        ) as cur:
            if await cur.fetchone():
                return False

    if _STORE is None or _STORE.embeddings.shape[0] == 0:
        await build_index()

    embedding = await encode_message(content)
    if _STORE is not None and _STORE.embeddings.shape[0]:
        distances = np.linalg.norm(_STORE.embeddings - embedding, axis=1)
        if distances.size and float(distances.min()) < threshold:
            return False

    await persist_embedding(content, embedding, tag, fingerprint)
    _add_to_index(content, embedding)
    _add_to_graph(content, "message", tag, embedding)
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
            await conn.execute(
                'INSERT OR IGNORE INTO concepts(name) VALUES (?)',
                (ent,),
            )
        for subj, rel, obj in relations:
            async with conn.execute(
                'SELECT id FROM concepts WHERE name = ?', (subj,)
            ) as cur:
                subj_id = await cur.fetchone()
            async with conn.execute(
                'SELECT id FROM concepts WHERE name = ?', (obj,)
            ) as cur:
                obj_id = await cur.fetchone()
            if subj_id and obj_id:
                await conn.execute(
                    'INSERT INTO relations(source, target, relation) VALUES (?, ?, ?)',
                    (subj_id[0], obj_id[0], rel),
                )
        await conn.commit()


def _token_trigrams(text: str) -> Set[str]:
    """Return a set of token trigrams from ``text``."""
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < 3:
        return set(tokens)
    return {
        " ".join(tokens[i : i + 3])
        for i in range(len(tokens) - 2)
    }


def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Compute Jaccard similarity between two trigram sets."""
    if not a and not b:
        return 1.0
    intersection = a & b
    union = a | b
    return len(intersection) / len(union) if union else 0.0


async def is_unique(sentence: str, threshold: float = 0.8) -> bool:
    """Return ``True`` if ``sentence`` is not similar to stored responses."""
    candidate_trigrams = _token_trigrams(sentence)
    async with get_connection() as conn:
        async with conn.execute('SELECT content FROM responses') as cur:
            rows = await cur.fetchall()
    for (content,) in rows:
        existing_trigrams = _token_trigrams(content)
        if _jaccard(candidate_trigrams, existing_trigrams) >= threshold:
            return False
    return True


async def store_response(sentence: str, tag: Optional[str] = None) -> None:
    """Persist a generated response."""
    async with get_connection() as conn:
        await conn.execute(
            'INSERT INTO responses(content, tag) VALUES (?, ?)',
            (sentence, tag),
        )
        await conn.commit()
    _add_to_graph(sentence, "response", tag)


async def fetch_recent(limit: int = 5) -> Tuple[List[str], List[str]]:
    """Fetch recent messages and responses from the hypergraph."""

    messages: List[str] = []
    responses: List[str] = []
    for content, kind in _MESSAGES[-limit*2:]:
        if kind == "message":
            messages.append(content)
        elif kind == "response":
            responses.append(content)
    return messages[-limit:], responses[-limit:]


async def fetch_recent_messages(limit: int = 50) -> List[Tuple[str, np.ndarray]]:
    """Fetch recent messages with embeddings from the hypergraph."""

    ids = _GRAPH.trail(limit * 2)
    results: List[Tuple[str, np.ndarray]] = []
    for node_id in ids:
        node = _GRAPH.get_node(node_id)
        if node and node.data.get("kind") == "message" and "embedding" in node.data:
            results.append((node.data["content"], node.data["embedding"]))
    return results[-limit:]


async def fetch_similar_messages(query: str, top_k: int = 5) -> List[str]:
    """Return top-k stored messages most similar to the query."""
    if _STORE is None or _STORE.embeddings.shape[0] == 0:
        await build_index()
    if _STORE is None or _STORE.embeddings.shape[0] == 0:
        return []
    q_vec = await encode_message(query)
    return _STORE.most_similar(q_vec, top_k, "global")


async def fetch_related_concepts(words: List[str]) -> List[str]:
    """Return relation sentences touching any of the given words."""
    results: List[str] = []
    seen = set()
    terms = [w.lower() for w in words]
    async with get_connection() as conn:
        for term in terms:
            async with conn.execute(
                (
                    "SELECT c1.name, r.relation, c2.name FROM relations r "
                    "JOIN concepts c1 ON r.source = c1.id "
                    "JOIN concepts c2 ON r.target = c2.id "
                    "WHERE LOWER(c1.name) = ? OR LOWER(c2.name) = ?"
                ),
                (term, term),
            ) as cur:
                rows = await cur.fetchall()
            for s, rel, t in rows:
                sentence = f"{s} {rel} {t}"
                if sentence not in seen:
                    seen.add(sentence)
                    results.append(sentence)
    return results
