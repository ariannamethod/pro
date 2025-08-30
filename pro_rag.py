from typing import Dict, List, Optional

import asyncio
import math
import json
from urllib import request, parse

from pro_metrics import tokenize, lowercase
import pro_memory
from memory.memory_lattice import MemoryGraphStore
import pro_predict


def _sentence_vector(words: List[str]) -> Dict[str, float]:
    vec: Dict[str, float] = {}
    for w in words:
        wvec = pro_predict._VECTORS.get(w)
        if not wvec:
            continue
        for k, v in wvec.items():
            vec[k] = vec.get(k, 0.0) + v
    return vec


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _external_search_sync(query: str, source: str, limit: int) -> List[str]:
    """Perform a blocking search against an external knowledge source."""
    if not query:
        return []
    if source == "wikipedia":
        url = "https://en.wikipedia.org/w/api.php?" + parse.urlencode(
            {
                "action": "opensearch",
                "search": query,
                "limit": str(limit),
                "namespace": "0",
                "format": "json",
            }
        )
        try:
            with request.urlopen(url, timeout=5) as resp:
                data = json.load(resp)
            return [d for d in data[2] if d]
        except Exception:
            return []
    return []


async def retrieve_external(
    query: str, source: str = "wikipedia", limit: int = 3
) -> List[str]:
    """Asynchronously retrieve information from an external storage."""
    return await asyncio.to_thread(_external_search_sync, query, source, limit)


async def retrieve(
    query_words: List[str],
    limit: int = 5,
    external_source: str | None = None,
    external_limit: int = 3,
    lattice: Optional[MemoryGraphStore] = None,
) -> List[str]:
    """Retrieve context using graph links and embedding similarity."""

    await asyncio.to_thread(pro_predict._ensure_vectors)
    qwords = lowercase(query_words)
    qvec = _sentence_vector(qwords)
    scored: List[tuple[float, str]] = []

    if lattice is not None:
        # Use embeddings stored in the lattice and include neighbouring nodes
        for did, nodes in lattice.graph.items():
            for idx, node in enumerate(nodes):
                sim = _cosine(qvec, node.embedding) if qvec else 0.0
                if sim <= 0:
                    continue
                scored.append((sim, node.text))
                # incorporate structural neighbours from the dialogue graph
                if idx > 0:
                    scored.append((sim * 0.5, nodes[idx - 1].text))
                if idx + 1 < len(nodes):
                    scored.append((sim * 0.5, nodes[idx + 1].text))
        scored.sort(key=lambda x: x[0], reverse=True)
    else:
        # Fall back to recent messages from the database
        messages = await pro_memory.fetch_recent_messages(50)
        qset = set(qwords)
        for msg, _ in messages:
            words = lowercase(tokenize(msg))
            word_score = len(qset.intersection(words))
            mvec = _sentence_vector(words)
            score = word_score + (_cosine(qvec, mvec) if qvec and mvec else 0)
            if score > 0:
                scored.append((score, msg))
        scored.sort(key=lambda x: x[0], reverse=True)

    graph_context = await pro_memory.fetch_related_concepts(qwords)
    external: List[str] = []
    if external_source:
        external = await retrieve_external(
            " ".join(qwords), external_source, external_limit
        )

    combined = external + graph_context + [m for _, m in scored]
    # Deduplicate while preserving order
    seen = set()
    result: List[str] = []
    for msg in combined:
        if msg in seen:
            continue
        seen.add(msg)
        result.append(msg)
        if len(result) >= limit:
            break
    return result


# ---------------------------------------------------------------------------
# Training demo
def _demo_training() -> None:
    """Small demonstration of online updates for ``ReinforceRetriever``.

    The demo sets up a tiny dialogue with two memories and repeatedly
    queries the retriever. Rewards favour the memory mentioning "pizza"
    which nudges the policy weights toward that node over time. The
    printed output shows the chosen memory and resulting reward for each
    step.
    """

    from memory import MemoryGraphStore, ReinforceRetriever

    store = MemoryGraphStore()
    did = "demo"
    store.add_utterance(did, "user", "hello world")
    store.add_utterance(did, "user", "pizza is tasty")
    retriever = ReinforceRetriever(store)

    for step in range(5):
        retriever.retrieve(did, "user")
        # Determine reward based on which message was sampled
        messages = [
            n.text for n in store.get_dialogue(did) if n.speaker == "user"
        ]
        idx = retriever._last[2] if retriever._last else 0
        text = messages[idx]
        reward = 1.0 if "pizza" in text else -1.0
        retriever.update(reward)
        print(f"step {step}: chose {text!r} reward={reward}")


if __name__ == "__main__":
    _demo_training()
