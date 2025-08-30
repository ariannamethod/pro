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
