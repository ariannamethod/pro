import asyncio
import numpy as np

# Simple deterministic "mini-SIAMESE" style embedding generator.
# This is intentionally lightweight so tests do not require heavy models.
_DIM = 32
_RNG = np.random.default_rng(0)
_W = (
    _RNG.normal(size=(256, _DIM)) + 1j * _RNG.normal(size=(256, _DIM))
).astype(np.complex64)
_W.flags.writeable = False


def _char_vector(text: str) -> np.ndarray:
    vec = np.zeros(256, dtype=np.float32)
    for ch in text.lower():
        vec[ord(ch) % 256] += 1.0
    return vec


async def embed_sentence(text: str) -> np.ndarray:
    """Return a normalized embedding for given text."""
    vec = _char_vector(text).astype(np.complex64)
    return await asyncio.to_thread(_project, vec)


def _project(vec: np.ndarray) -> np.ndarray:
    emb = vec @ _W
    norm = np.linalg.norm(emb)
    if norm:
        emb = emb / norm
    return emb.astype(np.complex64)


async def extract_entities_relations(text: str):
    """Naively extract entities and relations from a description.

    This lightweight helper looks for patterns of the form
    ``<subject> <verb> <object>`` using a small set of linking verbs.
    It returns a tuple ``(entities, relations)`` where ``entities`` is a
    list of unique concept names and ``relations`` is a list of
    ``(subject, verb, object)`` triples.
    """

    words = text.strip().split()
    lowered = [w.lower() for w in words]
    verbs = ["is", "are", "has", "have"]
    entities: list[str] = []
    relations: list[tuple[str, str, str]] = []

    for verb in verbs:
        if verb in lowered:
            idx = lowered.index(verb)
            subject = " ".join(words[:idx]).strip()
            obj = " ".join(words[idx + 1 :]).strip()
            if subject and obj:
                entities.extend([subject, obj])
                relations.append((subject, verb, obj))
            break

    if not entities and text.strip():
        entities = [text.strip()]

    # deduplicate while preserving order
    seen = set()
    unique_entities = []
    for ent in entities:
        if ent not in seen:
            seen.add(ent)
            unique_entities.append(ent)

    return unique_entities, relations
