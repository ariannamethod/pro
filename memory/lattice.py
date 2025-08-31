"""Concept lattice utilities extracted from pro_memory."""

from __future__ import annotations

import asyncio
from typing import List

import pro_rag_embedding

from .pooling import get_connection


async def add_concept(description: str) -> None:
    """Extract entities and relations from text and store them."""
    entities, relations = await pro_rag_embedding.extract_entities_relations(description)
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
