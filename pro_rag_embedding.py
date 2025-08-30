import numpy as np
import asyncio

# Simple deterministic "mini-SIAMESE" style embedding generator.
# This is intentionally lightweight so tests do not require heavy models.
_DIM = 32
_W = np.random.default_rng(0).normal(size=(256, _DIM)).astype(np.float32)
_LOCK = asyncio.Lock()


def _char_vector(text: str) -> np.ndarray:
    vec = np.zeros(256, dtype=np.float32)
    for ch in text.lower():
        vec[ord(ch) % 256] += 1.0
    return vec


async def embed_sentence(text: str) -> np.ndarray:
    """Return a normalized embedding for given text."""
    async with _LOCK:
        # embedding = char histogram projected via fixed random matrix
        vec = _char_vector(text)
        emb = vec @ _W
        norm = np.linalg.norm(emb)
        if norm:
            emb = emb / norm
        return emb.astype(np.float32)
