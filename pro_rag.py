from typing import Dict, List

import asyncio
import math

from pro_metrics import tokenize, lowercase
import pro_memory
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


async def retrieve(query_words: List[str], limit: int = 5) -> List[str]:
    """Retrieve context combining messages and concept relations."""
    await asyncio.to_thread(pro_predict._ensure_vectors)
    messages = await pro_memory.fetch_recent_messages(50)
    qwords = lowercase(query_words)
    qset = set(qwords)
    qvec = _sentence_vector(qwords)
    scored = []
    for msg, _ in messages:
        words = lowercase(tokenize(msg))
        word_score = len(qset.intersection(words))
        mvec = _sentence_vector(words)
        if qvec and mvec:
            score = word_score + _cosine(qvec, mvec)
        else:
            score = word_score
        if score > 0:
            scored.append((score, msg))
    scored.sort(reverse=True)
    graph_context = await pro_memory.fetch_related_concepts(qwords)
    combined = graph_context + [m for _, m in scored]
    return combined[:limit]


# ---------------------------------------------------------------------------
# Training demo
def _demo_training() -> None:
    """Small demonstration of online updates for ``ReinforceRetriever``.

    The demo sets up a tiny dialogue with two memories and repeatedly queries
    the retriever.  Rewards favour the memory mentioning "pizza" which nudges
    the policy weights toward that node over time.  The printed output shows the
    chosen memory and resulting reward for each step.
    """

    from memory import MemoryGraphStore, ReinforceRetriever

    store = MemoryGraphStore()
    did = "demo"
    store.add_utterance(did, "user", "hello world")
    store.add_utterance(did, "user", "pizza is tasty")
    retriever = ReinforceRetriever(store)

    for step in range(5):
        vec = retriever.retrieve(did, "user")
        # Determine reward based on which message was sampled
        messages = [n.text for n in store.get_dialogue(did) if n.speaker == "user"]
        idx = retriever._last[2] if retriever._last else 0
        text = messages[idx]
        reward = 1.0 if "pizza" in text else -1.0
        retriever.update(reward)
        print(f"step {step}: chose {text!r} reward={reward}")


if __name__ == "__main__":
    _demo_training()
