import os
import sqlite3
import asyncio
import threading
import pickle
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple
import math
import difflib
import numpy as np
import contextlib

import morphology
# Transformer блоки удалены - оставляем только n-gram логику

from pro_metrics import tokenize, lowercase
from compat import to_thread
from pro_memory import DB_PATH
import pro_memory
from metrics.timing import timed

TRANSFORMER_PATH = "pro_transformer.npz"

_GRAPH: Dict[str, Counter] = {}
_VECTORS: Dict[str, Dict[str, float]] = {}
_SYNONYMS: Dict[str, str] = {}
_LOCK = threading.RLock()
_SAVE_TASK: Optional[asyncio.Task] = None
_SAVE_WORKER: Optional[asyncio.Task] = None
_SAVE_QUEUE: Optional[asyncio.Queue] = None
_INIT_TASK: Optional[asyncio.Task] = None


@contextlib.contextmanager
def _vector_lock() -> None:
    """Synchronise access to the shared embedding structures."""
    _LOCK.acquire()
    try:
        yield
    finally:
        _LOCK.release()


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
    files: List[str] = []
    main_path = os.path.join(dataset_dir, "smalltalk.txt")
    if os.path.isfile(main_path):
        files.append(main_path)
    for name in os.listdir(dataset_dir):
        if name == "smalltalk.txt" or name.endswith(".pkl"):
            continue
        path = os.path.join(dataset_dir, name)
        if not os.path.isfile(path):
            continue
        files.append(path)
    for path in files:
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


async def _ensure_vectors() -> None:
    """Initialise the global co-occurrence graph and vectors.

    Heavy file I/O and graph construction are executed in a thread so that
    callers do not block the event loop. Only the shared state assignment is
    protected by the vector lock.
    """

    global _GRAPH, _VECTORS

    # Fast path – already initialised.
    with _vector_lock():
        if _VECTORS:
            return

    try:
        graph, vectors = await to_thread(load_embeddings)
        if not vectors:
            raise ValueError("empty embeddings")
    except Exception:
        graph = await to_thread(_build_graph)
        vectors = await to_thread(_build_embeddings, graph)
        try:
            await to_thread(save_embeddings, graph, vectors)
        except Exception:
            pass

    with _vector_lock():
        if not _VECTORS:
            _GRAPH, _VECTORS = graph, vectors


def start_background_init() -> None:
    """Kick off vector initialisation in the background."""
    global _INIT_TASK
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    if _INIT_TASK is None or _INIT_TASK.done():
        _INIT_TASK = loop.create_task(_ensure_vectors())


def _log_save_error(task: asyncio.Task) -> None:
    try:
        exc = task.exception()
        if exc:
            logging.error("Saving embeddings failed: %s", exc)
    except asyncio.CancelledError:
        pass


async def wait_save_task() -> None:
    global _SAVE_TASK, _SAVE_QUEUE
    if _SAVE_QUEUE is not None:
        await _SAVE_QUEUE.join()
    if _SAVE_TASK is not None:
        try:
            await _SAVE_TASK
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logging.error("Saving embeddings failed: %s", exc)
        _SAVE_TASK = None


async def _save_worker() -> None:
    global _SAVE_TASK
    assert _SAVE_QUEUE is not None
    while True:
        await _SAVE_QUEUE.get()
        _SAVE_QUEUE.task_done()
        if _SAVE_TASK is not None:
            try:
                await _SAVE_TASK
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logging.error("Saving embeddings failed: %s", exc)
        _SAVE_TASK = asyncio.create_task(
            to_thread(save_embeddings, _GRAPH, _VECTORS)
        )
        _SAVE_TASK.add_done_callback(_log_save_error)


@timed
async def update(word_list: List[str]) -> None:
    """Update the co-occurrence graph and vectors with new words.

    The *word_list* should contain individual tokens. After the update the
    words become part of the internal vocabulary used by :func:`suggest`.
    """
    global _VECTORS, _SAVE_QUEUE, _SAVE_WORKER
    await _ensure_vectors()
    with _vector_lock():
        words = lowercase(word_list)
        for i, word in enumerate(words):
            for j in range(i + 1, len(words)):
                other = words[j]
                if not word or not other:
                    continue
                _GRAPH.setdefault(word, Counter())[other] += 1
                _GRAPH.setdefault(other, Counter())[word] += 1
        _VECTORS = _build_embeddings(_GRAPH)
    loop = asyncio.get_running_loop()
    if _SAVE_QUEUE is None or _SAVE_WORKER is None or _SAVE_WORKER.done():
        _SAVE_QUEUE = asyncio.Queue()
        _SAVE_WORKER = loop.create_task(_save_worker())
    await _SAVE_QUEUE.put(None)


TOKENS_QUEUE: Optional[asyncio.Queue] = None
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
        await wait_save_task()
        batch.clear()


async def enqueue_tokens(tokens: List[str]) -> None:
    """Add *tokens* to the update queue and ensure the worker runs."""
    global TOKENS_QUEUE, _QUEUE_LOOP, _UPDATE_TASK
    if _INIT_TASK is None:
        start_background_init()
    if not _VECTORS:
        if _INIT_TASK is None or not _INIT_TASK.done():
            raise RuntimeError("vector initialisation in progress")
        _INIT_TASK.result()
        if not _VECTORS:
            raise RuntimeError("vector initialisation failed")
    loop = asyncio.get_running_loop()
    if TOKENS_QUEUE is None or _QUEUE_LOOP is not loop:
        TOKENS_QUEUE = asyncio.Queue()
        _QUEUE_LOOP = loop
        _UPDATE_TASK = loop.create_task(_update_worker())
    await TOKENS_QUEUE.put(tokens)


async def suggest_async(word: str, topn: int = 3) -> List[str]:
    """Return up to *topn* words semantically close to *word*.

    If *word* is known from the dataset, cosine similarity in the
    co-occurrence embedding space is used. For out-of-vocabulary words a
    fuzzy string match against the vocabulary is performed.
    """

    if _INIT_TASK is None:
        start_background_init()
    if not _VECTORS:
        if _INIT_TASK is None or not _INIT_TASK.done():
            return []
        try:
            _INIT_TASK.result()
        except Exception:
            return []
        if not _VECTORS:
            return []
    with _vector_lock():
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


def suggest(word: str, topn: int = 3) -> List[str]:
    """Synchronous wrapper around :func:`suggest_async`."""

    return asyncio.run(suggest_async(word, topn))


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

    def __init__(
        self,
        vocab: List[str],
        dim: int = 32,
        use_gate: bool = True,
        lr: float = 0.1,
        l2: float = 0.0,
        temperature: float = 1.0,
        clip_norm: float = 1.0,
        repeat_penalty: float = 1.0,
    ) -> None:
        self.vocab = vocab
        self.dim = dim
        rng = np.random.default_rng(0)
        self.emb = rng.standard_normal((len(vocab), dim))
        self.w_q = rng.standard_normal((dim, dim))
        self.w_k = rng.standard_normal((dim, dim))
        self.w_v = rng.standard_normal((dim, dim))
        self.w_o = rng.standard_normal((dim, len(vocab)))
        self.use_gate = use_gate
        # DynamicContextGate удален
        self.lr = lr
        self.l2 = l2
        self.temperature = temperature
        self.clip_norm = clip_norm
        self.repeat_penalty = repeat_penalty
        if os.path.exists(TRANSFORMER_PATH):
            try:
                data = np.load(TRANSFORMER_PATH, allow_pickle=True)
                file_vocab = list(data["vocab"])
                if file_vocab == vocab:
                    self.emb = data["emb"]
                    self.w_q = data["w_q"]
                    self.w_k = data["w_k"]
                    self.w_v = data["w_v"]
                    self.w_o = data["w_o"]
                    if self.gate and "gate_bias" in data:
                        self.gate.load_state_dict({"bias": data["gate_bias"]})
            except Exception:
                pass

    def save(self, path: str = TRANSFORMER_PATH) -> None:
        np.savez(
            path,
            vocab=np.array(self.vocab),
            emb=self.emb,
            w_q=self.w_q,
            w_k=self.w_k,
            w_v=self.w_v,
            w_o=self.w_o,
            gate_bias=np.zeros(self.dim),  # gate удален
        )

    def train_step(
        self,
        tokens_batch: List[List[str]],
        targets: List[str],
        lr: Optional[float] = None,
    ) -> None:
        if lr is not None:
            self.lr = lr
        batch_ids = [
            [self.vocab.index(t) for t in tokens if t in self.vocab]
            for tokens in tokens_batch
        ]
        valid = [ids and target in self.vocab for ids, target in zip(batch_ids, targets)]
        if not any(valid):
            return
        max_len = max(len(ids) for ids, v in zip(batch_ids, valid) if v)
        batch_size = len(tokens_batch)
        ids_arr = -np.ones((batch_size, max_len), dtype=int)
        mask = np.zeros((batch_size, max_len), dtype=bool)
        for i, ids in enumerate(batch_ids):
            if not ids:
                continue
            ids_arr[i, : len(ids)] = ids
            mask[i, : len(ids)] = True
        x = self.emb[ids_arr.clip(min=0)]
        x[~mask] = 0
        q = x @ self.w_q
        k = x @ self.w_k
        v = x @ self.w_v
        scale = np.sqrt(self.dim)
        att = q @ np.swapaxes(k, 1, 2) / scale
        att = np.where(mask[:, :, None] & mask[:, None, :], att, -np.inf)
        att = np.exp(att - np.max(att, axis=-1, keepdims=True))
        att_sum = att.sum(axis=-1, keepdims=True)
        att = np.divide(att, att_sum, where=att_sum != 0)
        context = att @ v
        # quantum_dropout удален
        # gate удален
        context = (context - context.mean(axis=-1, keepdims=True)) / (
            context.std(axis=-1, keepdims=True) + 1e-5
        )
        pooled = context.mean(axis=1)
        out = pooled @ self.w_o
        out /= self.temperature
        exp_out = np.exp(out - np.max(out, axis=1, keepdims=True))
        probs = exp_out / exp_out.sum(axis=1, keepdims=True)
        y = np.zeros_like(probs)
        for i, target in enumerate(targets):
            if valid[i]:
                y[i, self.vocab.index(target)] = 1.0
        grad = probs - y
        grad_w_o = pooled.T @ grad / batch_size + self.l2 * self.w_o
        norm = np.linalg.norm(grad_w_o)
        if norm > self.clip_norm:
            grad_w_o *= self.clip_norm / (norm + 1e-6)
        self.w_o -= self.lr * grad_w_o

    def logits(
        self,
        tokens: List[str],
        adapters: Optional[List[Dict[str, float]]] = None,
    ) -> Dict[str, float]:
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
        # quantum_dropout и gate удалены
        context = (context - context.mean(axis=-1, keepdims=True)) / (
            context.std(axis=-1, keepdims=True) + 1e-5
        )
        pooled = context.mean(axis=0)
        out = pooled @ self.w_o
        counts = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        for token, c in counts.items():
            if c > 1 and token in self.vocab:
                out[self.vocab.index(token)] -= self.repeat_penalty * (c - 1)
        out /= self.temperature
        exp_out = np.exp(out - out.max())
        probs = exp_out / exp_out.sum()
        logits = {self.vocab[i]: float(probs[i]) for i in range(len(self.vocab))}
        if adapters:
            for adapter in adapters:
                for word, bias in adapter.items():
                    if word in logits:
                        logits[word] += bias
        return logits


_TRANSFORMERS: Dict[tuple, MiniSelfAttention] = {}


def transformer_logits(
    tokens: List[str],
    vocab: List[str],
    adapters: Optional[List[Dict[str, float]]] = None,
) -> Dict[str, float]:
    """Return next-word logits for *tokens* using a tiny transformer."""
    key = tuple(vocab)
    if key not in _TRANSFORMERS:
        _TRANSFORMERS[key] = MiniSelfAttention(vocab)
    model = _TRANSFORMERS[key]
    return model.logits(tokens, adapters=adapters)


async def update_transformer(
    vocab: List[str],
    messages: Optional[List[str]] = None,
    responses: Optional[List[str]] = None,
) -> None:
    """Train the transformer using recent message/response pairs."""
    if messages is None or responses is None:
        messages, responses = await pro_memory.fetch_recent(20)
    if not messages:
        return
    key = tuple(vocab)
    if key not in _TRANSFORMERS:
        _TRANSFORMERS[key] = MiniSelfAttention(vocab)
    model = _TRANSFORMERS[key]
    pairs: List[Tuple[List[str], str]] = []
    for msg, resp in zip(messages, responses):
        tokens = lowercase(tokenize(msg))[-5:]
        targets = lowercase(tokenize(resp))
        if not targets:
            continue
        pairs.append((tokens, targets[0]))
    if pairs:
        tokens_batch, targets = zip(*pairs)
        model.train_step(list(tokens_batch), list(targets))
        await to_thread(model.save)


def combine_predictions(
    ngram_pred: str,
    trans_logits: Dict[str, float],
    ngram_weight: float = 1.0,
    transformer_weight: float = 1.0,
) -> List[str]:
    """Combine n-gram and transformer predictions using weighted scores.

    The *ngram_pred* is given a fixed *ngram_weight* while each transformer
    logit in *trans_logits* is scaled by *transformer_weight*. The resulting
    unique words are returned sorted by descending score. At most the top two
    predictions are returned to mirror previous behaviour.
    """

    scores: Dict[str, float] = {}
    if ngram_pred:
        scores[ngram_pred] = scores.get(ngram_pred, 0.0) + ngram_weight
    for word, logit in (trans_logits or {}).items():
        scores[word] = scores.get(word, 0.0) + logit * transformer_weight
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [w for w, _ in ordered][:2]
