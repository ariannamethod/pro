import os
from collections import Counter, defaultdict
from typing import Dict, List
import math
import difflib
import random

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


def dissimilar(words: List[str], count: int) -> List[str]:
    """Return *count* words with low cosine similarity to *words*.

    Words already present in *words* are excluded. When insufficient
    candidates are available from the embedding vocabulary the remaining
    slots are filled with random vocabulary terms.
    """

    _ensure_vectors()
    vocab = set(_VECTORS.keys())
    candidates = list(vocab - set(words))
    if not candidates:
        return []
    scores: Dict[str, float] = {}
    for cand in candidates:
        vec = _VECTORS.get(cand)
        if not vec:
            scores[cand] = 0.0
            continue
        max_sim = 0.0
        for w in words:
            wvec = _VECTORS.get(w)
            if not wvec:
                continue
            keys = set(vec) | set(wvec)
            dot = sum(vec.get(k, 0.0) * wvec.get(k, 0.0) for k in keys)
            norm_a = math.sqrt(sum(v * v for v in vec.values()))
            norm_b = math.sqrt(sum(v * v for v in wvec.values()))
            if norm_a == 0 or norm_b == 0:
                continue
            sim = dot / (norm_a * norm_b)
            if sim > max_sim:
                max_sim = sim
        scores[cand] = max_sim
    ordered = sorted(scores.items(), key=lambda x: x[1])
    selected = [w for w, _ in ordered[:count]]
    if len(selected) < count:
        remaining = [w for w in candidates if w not in selected]
        random.shuffle(remaining)
        selected.extend(remaining[: count - len(selected)])
    return selected
