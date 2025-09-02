from compat import to_thread
import asyncio
from typing import Dict, Iterable

import grammar_filters
import pro_memory
import pro_predict


async def build_analog_map(tokens: Iterable[str]) -> Dict[str, str]:
    """Return mapping of tokens to analog replacements.

    For each token the function first tries :func:`pro_predict.suggest_async` and
    falls back to :func:`pro_predict.lookup_analogs`.
    """
    analog_map: Dict[str, str] = {}
    lock = asyncio.Lock()

    async def _build(tok: str) -> None:
        suggestions = await pro_predict.suggest_async(tok, topn=1)
        analog = suggestions[0] if suggestions else None
        if not analog:
            analog = await to_thread(pro_predict.lookup_analogs, tok)
        if analog:
            async with lock:
                analog_map[tok] = analog

    await asyncio.gather(*(_build(tok) for tok in tokens))
    return analog_map


async def ensure_unique(response: str) -> bool:
    """Persist ``response`` if it passes grammar filters and is novel.

    Returns ``True`` if the response was stored, ``False`` otherwise.
    """
    if grammar_filters.passes_filters(response) and await pro_memory.is_unique(response):
        await pro_memory.store_response(response)
        return True
    return False
