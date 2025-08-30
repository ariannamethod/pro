import os
import sqlite3
import asyncio
import pickle
from collections import Counter, defaultdict
from typing import Dict, List, Optional
import math
import difflib
import numpy as np

import morphology

from pro_metrics import tokenize, lowercase
from pro_memory import DB_PATH

_GRAPH: Dict[str, Counter] = {}
_VECTORS: Dict[str, Dict[str, float]] = {}
_SYNONYMS: Dict[str, str] = {}

if os.path.exists(DB_PATH):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT word, analog FROM synonyms")
        _SYNONYMS = {w: a for w, a in cur.fetchall() if w and a}
        conn.close()
    except sqlite3.Error:
        _SYNONYMS = {}


def _build_graph(dataset_dir: str = "datasets") -> Dict[str, Counter]:
    """Build a simple co-occurrence graph from dataset files."""
    graph: Dict[str, Counter] = defaultdict(Counter)
    if not os.path.exists(dataset_dir):
        return graph
    for name in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, name)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                words = lowercase(tokenize(line))
                for i, word in enumerate(words):
                    for j in range(i + 1, len(words)):
                        other = words[j]
                        if not word or not other:
                            continue
                        graph[word][other] += 1
                        graph[other][word] += 1
    return graph


def _build_embeddings(
    graph: Dict[str, Counter],
) -> Dict[str, Dict[str, float]]:
    """Convert co-occurrence graph to normalised frequency vectors."""
    vectors: Dict[str, Dict[str, float]] = {}
    for word, neighbours in graph.items():
        total = sum(neighbours.values()) or 1
        vectors[word] = {n: cnt / total for n, cnt in neighbours.items()}
    return vectors


def save_embeddings(
    graph: Dict[str, Counter],
    vectors: Dict[str, Dict[str, float]],
    path: str = "datasets/embeddings.pkl",
) -> None:
    """Persist *graph* and *vectors* to *path* using pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump((graph, vectors), fh)


def load_embeddings(
    path: str = "datasets/embeddings.pkl",
) -> Optional[tuple]:
    """Load *graph* and *vectors* from *path* if it exists."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _ensure_vectors() -> None:
    global _GRAPH, _VECTORS
    if _VECTORS:
        return
    try:
        _GRAPH, _VECTORS = load_embeddings()
        if not _VECTORS:
            raise ValueError("empty embeddings")
    except Exception:
        _GRAPH = _build_graph()
        _VECTORS = _build_embeddings(_GRAPH)
        try:
            save_embeddings(_GRAPH, _VECTORS)
        except Exception:
            pass


async def update(word_list: List[str]) -> None:
    """Update the co-occurrence graph and vectors with new words.

    The *word_list* should contain individual tokens. After the update the
    words become part of the internal vocabulary used by :func:`suggest`.
    """
    global _VECTORS
    _ensure_vectors()
    words = lowercase(word_list)
    for i, word in enumerate(words):
        for j in range(i + 1, len(words)):
            other = words[j]
            if not word or not other:
                continue
            _GRAPH.setdefault(word, Counter())[other] += 1
            _GRAPH.setdefault(other, Counter())[word] += 1
    _VECTORS = _build_embeddings(_GRAPH)
    asyncio.create_task(asyncio.to_thread(save_embeddings, _GRAPH, _VECTORS))


TOKENS_QUEUE: Optional[asyncio.Queue[List[str]]] = None
_QUEUE_LOOP: Optional[asyncio.AbstractEventLoop] = None
_UPDATE_TASK: Optional[asyncio.Task] = None


async def _update_worker() -> None:
    """Background task that flushes queued tokens in batches."""
    batch: List[str] = []
    while True:
        assert TOKENS_QUEUE is not None
        items = await TOKENS_QUEUE.get()
        batch.extend(items)
        TOKENS_QUEUE.task_done()
        # Drain any other pending items without waiting
        while not TOKENS_QUEUE.empty():
            more = await TOKENS_QUEUE.get()
            batch.extend(more)
            TOKENS_QUEUE.task_done()
        await update(batch)
        batch.clear()


async def enqueue_tokens(tokens: List[str]) -> None:
    """Add *tokens* to the update queue and ensure the worker runs."""
    global TOKENS_QUEUE, _QUEUE_LOOP, _UPDATE_TASK
    loop = asyncio.get_running_loop()
    if TOKENS_QUEUE is None or _QUEUE_LOOP is not loop:
        TOKENS_QUEUE = asyncio.Queue()
        _QUEUE_LOOP = loop
        _UPDATE_TASK = loop.create_task(_update_worker())
    await TOKENS_QUEUE.put(tokens)


def suggest(word: str, topn: int = 3) -> List[str]:
    """Return up to *topn* words semantically close to *word*.

    If *word* is known from the dataset, cosine similarity in the
    co-occurrence embedding space is used. For out-of-vocabulary words a
    fuzzy string match against the vocabulary is performed.
    """
    _ensure_vectors()
    if word not in _GRAPH and word not in _VECTORS:
        return []
    neighbours = _GRAPH.get(word)
    if neighbours:
        return [w for w, _ in neighbours.most_common(topn)]
    vec = _VECTORS.get(word)
    if not vec:
        return []
    scores: Dict[str, float] = {}
    for other, ovec in _VECTORS.items():
        if other == word:
            continue
        keys = set(vec) | set(ovec)
        dot = sum(vec.get(k, 0.0) * ovec.get(k, 0.0) for k in keys)
        norm_a = math.sqrt(sum(v * v for v in vec.values()))
        norm_b = math.sqrt(sum(v * v for v in ovec.values()))
        if norm_a == 0 or norm_b == 0:
            continue
        scores[other] = dot / (norm_a * norm_b)
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in ordered[:topn]]


def lookup_analogs(word: str) -> Optional[str]:
    """Return a known analog for *word* or a suggestion.

    The function first checks an in-memory cache populated from the
    ``synonyms`` table in ``pro_memory.db`` at import time. If no entry is
    found, it falls back to :func:`suggest` and returns the best match if
    available.
    """

    analog = _SYNONYMS.get(word)
    if analog:
        return analog

    suggestions = suggest(word, topn=1)
    if suggestions:
        cand = suggestions[0]
        if difflib.SequenceMatcher(None, word, cand).ratio() >= 0.95:
            return cand

    # Try to find an analog for the root and rebuild the word using known
    # prefixes/suffixes.  The morphological analysis is cached in
    # :mod:`morphology` so the extra work is cheap on repeated calls.
    root, prefixes, suffixes = morphology.split(word)
    if root != word:
        analog_root = lookup_analogs(root)
        if analog_root:
            return "".join(prefixes) + analog_root + "".join(suffixes)

    return None


class MiniSelfAttention:
    """A tiny self-attention module for next-word prediction."""

    def __init__(self, vocab: List[str], dim: int = 32) -> None:
        self.vocab = vocab
        self.dim = dim
        rng = np.random.default_rng(0)
        self.emb = rng.standard_normal((len(vocab), dim))
        self.w_q = rng.standard_normal((dim, dim))
        self.w_k = rng.standard_normal((dim, dim))
        self.w_v = rng.standard_normal((dim, dim))
        self.w_o = rng.standard_normal((dim, len(vocab)))

    def logits(self, tokens: List[str]) -> Dict[str, float]:
        ids = [self.vocab.index(t) for t in tokens if t in self.vocab]
        if not ids:
            return {w: 0.0 for w in self.vocab}
        x = self.emb[ids]
        q = x @ self.w_q
        k = x @ self.w_k
        v = x @ self.w_v
        scale = np.sqrt(self.dim)
        att = q @ k.T / scale
        att = np.exp(att - att.max(axis=-1, keepdims=True))
        att = att / att.sum(axis=-1, keepdims=True)
        context = att @ v
        pooled = context.mean(axis=0)
        out = pooled @ self.w_o
        return {self.vocab[i]: float(out[i]) for i in range(len(self.vocab))}


_TRANSFORMERS: Dict[tuple, MiniSelfAttention] = {}


def transformer_logits(tokens: List[str], vocab: List[str]) -> Dict[str, float]:
    """Return next-word logits for *tokens* using a tiny transformer."""
    key = tuple(vocab)
    if key not in _TRANSFORMERS:
        _TRANSFORMERS[key] = MiniSelfAttention(vocab)
    model = _TRANSFORMERS[key]
    return model.logits(tokens)
