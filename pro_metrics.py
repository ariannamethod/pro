import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"\b\w+\b")


def tokenize(text: str):
    """Split text into lowercase tokens."""
    return TOKEN_RE.findall(text.lower())


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


def compute_metrics(words, bigram_counts, word_counts):
    return {
        "entropy": entropy(words),
        "perplexity": perplexity(words, bigram_counts, word_counts),
        "resonance": resonance(words, bigram_counts),
    }
