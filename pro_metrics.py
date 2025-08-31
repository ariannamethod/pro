import math
import re
import logging
from collections import Counter, deque
from typing import Deque, Dict, List, Tuple

TOKEN_RE = re.compile(r"\b\w+\b")


def tokenize(text: str):
    """Split text into tokens without altering case."""
    return TOKEN_RE.findall(text)


def lowercase(words):
    """Return a lowercase copy of tokens for case-insensitive metrics."""
    return [w.lower() for w in words]


def entropy(words):
    """Shannon entropy of token distribution."""
    if not words:
        return 0.0
    counts = Counter(words)
    total = len(words)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def perplexity(words, bigram_counts, word_counts):
    """Compute perplexity of words given a bigram model."""
    if not words:
        return 0.0
    vocab = len(word_counts) or 1
    log_prob = 0.0
    prev = "<s>"
    for word in words:
        prev_counts = bigram_counts.get(prev, {})
        numerator = prev_counts.get(word, 0) + 1
        denominator = word_counts.get(prev, 0) + vocab
        prob = numerator / denominator
        log_prob += -math.log(prob)
        prev = word
    return math.exp(log_prob / len(words))


def trigram_perplexity(
    words, trigram_counts: Dict[Tuple[str, str], Dict[str, int]], word_counts
):
    """Compute perplexity of words given a trigram model."""
    if not words:
        return 0.0
    vocab = len(word_counts) or 1
    log_prob = 0.0
    prev2 = "<s>"
    prev1 = "<s>"
    for word in words:
        prev_counts = trigram_counts.get((prev2, prev1), {})
        numerator = prev_counts.get(word, 0) + 1
        denominator = sum(prev_counts.values()) + vocab
        prob = numerator / denominator
        log_prob += -math.log(prob)
        prev2, prev1 = prev1, word
    return math.exp(log_prob / len(words))


def resonance(words, bigram_counts):
    """Simple resonance metric: average bigram count along the sequence."""
    if not words:
        return 0.0
    total = 0
    prev = "<s>"
    for word in words:
        total += bigram_counts.get(prev, {}).get(word, 0)
        prev = word
    return total / len(words)


def trigram_resonance(
    words, trigram_counts: Dict[Tuple[str, str], Dict[str, int]]
):
    """Average trigram count along the sequence."""
    if not words:
        return 0.0
    total = 0
    prev2 = "<s>"
    prev1 = "<s>"
    for word in words:
        total += trigram_counts.get((prev2, prev1), {}).get(word, 0)
        prev2, prev1 = prev1, word
    return total / len(words)


def char_ngram_resonance(words, char_counts, n: int = 3):
    """Average character n-gram count across words."""
    if not words:
        return 0.0
    total = 0
    cnt = 0
    for word in words:
        for i in range(len(word) - n + 1):
            ngram = word[i: i + n]
            total += char_counts.get(ngram, 0)
            cnt += 1
    return total / cnt if cnt else 0.0


def compute_metrics(
    words,
    trigram_counts,
    bigram_counts,
    word_counts,
    char_ngram_counts,
    char_n: int = 3,
):
    return {
        "entropy": entropy(words),
        "perplexity": perplexity(words, bigram_counts, word_counts),
        "resonance": resonance(words, bigram_counts),
        "trigram_perplexity": trigram_perplexity(
            words, trigram_counts, word_counts
        ),
        "trigram_resonance": trigram_resonance(words, trigram_counts),
        "char_ngram_resonance": char_ngram_resonance(
            words, char_ngram_counts, char_n
        ),
    }


def target_length_from_metrics(
    metrics: Dict[str, float], min_len: int = 6, max_len: int = 10
) -> int:
    """Map aggregated metric values to a length in ``[min_len, max_len]``."""
    span = max_len - min_len + 1
    total = sum(metrics.values())
    return min_len + int(total) % span


# ---------------------------------------------------------------------------
# Latency tracking

_LATENCIES: Dict[str, Deque[float]] = {}
_LAT_WINDOW = 100


def record_latency(
    name: str, duration: float, window: int = _LAT_WINDOW
) -> None:
    """Record a latency measurement for *name* keeping only the latest
    entries."""

    dq = _LATENCIES.setdefault(name, deque(maxlen=window))
    dq.append(duration)


def _percentile(sorted_data: List[float], p: float) -> float:
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    d0 = sorted_data[int(f)] * (c - k)
    d1 = sorted_data[int(c)] * (k - f)
    return d0 + d1


def latency_stats(name: str) -> Dict[str, float]:
    """Return average and percentile latency statistics for *name*."""

    data = list(_LATENCIES.get(name, []))
    if not data:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0}
    data.sort()
    avg = sum(data) / len(data)
    return {
        "avg": avg,
        "p50": _percentile(data, 0.50),
        "p95": _percentile(data, 0.95),
    }


def all_latency_stats() -> Dict[str, Dict[str, float]]:
    """Return latency statistics for all tracked metrics."""

    return {name: latency_stats(name) for name in _LATENCIES}


def format_latency_stats() -> List[str]:
    """Return human readable strings for all latency stats."""

    lines: List[str] = []
    for name, stats in all_latency_stats().items():
        lines.append(
            f"{name}: avg={stats['avg']:.3f}s "
            f"p50={stats['p50']:.3f}s "
            f"p95={stats['p95']:.3f}s"
        )
    return lines


def log_latency_stats() -> None:
    """Log latency statistics using the standard logger."""

    for line in format_latency_stats():
        logging.info(line)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Metrics utilities")
    parser.add_argument(
        "--latency", action="store_true", help="print latency statistics"
    )
    args = parser.parse_args()
    if args.latency:
        for line in format_latency_stats():
            print(line)
