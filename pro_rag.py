from typing import List

import numpy as np

from pro_metrics import tokenize, lowercase
import pro_memory
import pro_rag_embedding


async def retrieve(query_words: List[str], limit: int = 5) -> List[str]:
    """Retrieve relevant messages combining word overlap and embedding similarity."""
    messages = await pro_memory.fetch_recent_messages(50)
    qset = set(lowercase(query_words))
    query_emb = await pro_rag_embedding.embed_sentence(' '.join(query_words))
    scored = []
    for msg, emb in messages:
        words = lowercase(tokenize(msg))
        word_score = len(qset.intersection(words))
        cos_sim = float(np.dot(query_emb, emb))
        score = word_score + cos_sim
        if score > 0:
            scored.append((score, msg))
    scored.sort(reverse=True)
    return [m for _, m in scored[:limit]]
