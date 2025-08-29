import os
from collections import Counter, defaultdict
from typing import Dict, List
import math
import difflib
import numpy as np

from pro_metrics import tokenize, lowercase

_GRAPH: Dict[str, Counter] = {}
_VECTORS: Dict[str, Dict[str, float]] = {}


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


def _ensure_vectors() -> None:
    global _GRAPH, _VECTORS
    if _VECTORS:
        return
    _GRAPH = _build_graph()
    _VECTORS = _build_embeddings(_GRAPH)


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


def suggest(word: str, topn: int = 3) -> List[str]:
    """Return up to *topn* words semantically close to *word*.

    If *word* is known from the dataset, cosine similarity in the
    co-occurrence embedding space is used. For out-of-vocabulary words a
    fuzzy string match against the vocabulary is performed.
    """
    _ensure_vectors()
    if word in _VECTORS:
        vec = _VECTORS[word]
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
    vocab = list(_VECTORS.keys())
    return difflib.get_close_matches(word, vocab, n=topn)


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
