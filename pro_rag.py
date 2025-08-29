from typing import List
from pro_metrics import tokenize
import pro_memory

async def retrieve(query_words: List[str], limit: int = 5) -> List[str]:
    """Retrieve relevant messages from memory based on word overlap."""
    messages = await pro_memory.fetch_recent(50)
    qset = set(query_words)
    scored = []
    for msg in messages:
        words = tokenize(msg)
        score = len(qset.intersection(words))
        if score:
            scored.append((score, msg))
    scored.sort(reverse=True)
    return [m for _, m in scored[:limit]]
