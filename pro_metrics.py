import math
import re
from collections import Counter
from typing import Dict, Tuple

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
            ngram = word[i : i + n]
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
        "trigram_perplexity": trigram_perplexity(words, trigram_counts, word_counts),
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
