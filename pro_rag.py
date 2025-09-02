from typing import Dict, List, Optional, Tuple

import asyncio
import math
import os

import aiohttp

from pro_metrics import tokenize, lowercase
import pro_memory
# MemoryStore не нужен
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


_external_cache: Dict[Tuple[str, str, int, str], List[str]] = {}


async def retrieve_external(
    query: str, source: str = "wikipedia", limit: int = 3
) -> List[str]:
    """Asynchronously retrieve information from an external storage."""
    if not query:
        return []
    if source == "wikipedia":
        api_url = os.getenv(
            "WIKIPEDIA_API", "https://en.wikipedia.org/w/api.php"
        )
        cache_key = (source, query, limit, api_url)
        if cache_key in _external_cache:
            return _external_cache[cache_key]
        params = {
            "action": "opensearch",
            "search": query,
            "limit": str(limit),
            "namespace": "0",
            "format": "json",
        }
        timeout_val = float(os.getenv("RAG_EXTERNAL_TIMEOUT", "3"))
        timeout = aiohttp.ClientTimeout(total=timeout_val)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(api_url, params=params) as resp:
                    if resp.status != 200:
                        result: List[str] = []
                    else:
                        data = await resp.json()
                        result = [d for d in data[2] if d]
        except asyncio.TimeoutError:
            result = []
        except asyncio.CancelledError:
            raise
        except aiohttp.ClientError:
            result = []
        except Exception:
            result = []
        _external_cache[cache_key] = result
        return result
    return []


async def retrieve(
    query_words: List[str],
    limit: int = 5,
    external_source: Optional[str] = None,
    external_limit: int = 3,
    # lattice удален
) -> List[str]:
    """Retrieve context using graph links and embedding similarity."""

    await pro_predict._ensure_vectors()
    qwords = lowercase(query_words)
    qvec = _sentence_vector(qwords)
    scored: List[tuple[float, str]] = []

    # lattice удален - используем прямой поиск по памяти
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

    graph_task = asyncio.create_task(pro_memory.fetch_related_concepts(qwords))
    external: List[str] = []
    if external_source:
        external_task = asyncio.create_task(
            retrieve_external(
                " ".join(qwords), external_source, external_limit
            )
        )
        timeout_val = float(os.getenv("RAG_EXTERNAL_TIMEOUT", "3"))
        try:
            external = await asyncio.wait_for(
                external_task, timeout=timeout_val
            )
        except asyncio.TimeoutError:
            external_task.cancel()
            external = []
        except asyncio.CancelledError:
            external_task.cancel()
            raise
        except Exception:
            external = []
    graph_context = await graph_task

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

    # MemoryStore и ReinforceRetriever удалены

    # store удален
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
