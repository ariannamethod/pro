import logging
import re
from typing import Iterable

# Strictly forbidden patterns ---------------------------------------------

ARTICLE_PAIR_RE = re.compile(r"\b(the|a|an) (the|a|an)\b", re.IGNORECASE)
# article 'a' before prepositions with no subsequent word
A_PREP_RE = re.compile(r"\ba (in|on|to|by|at|of)\b(?! \w)", re.IGNORECASE)
DUP_WORD_RE = re.compile(r"\b(\w+)\b \1\b", re.IGNORECASE)
SINGLE_LETTER_PAIR_RE = re.compile(r"\b\w\b \b\w\b")

DUP_WHITELIST = {"go", "no", "yeah"}
VERB_SET = {
    "is",
    "am",
    "are",
    "was",
    "were",
    "be",
    "being",
    "been",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "will",
    "would",
    "can",
    "could",
    "should",
    "shall",
    "may",
    "might",
    "must",
}
SINGLE_PAIR_WHITELIST = {
    "i am",
    "i go",
    "i do",
    "i can",
    "i see",
    "i will",
}

# High entropy patterns ---------------------------------------------------
SENTENCE_END_PREP_RE = re.compile(
    r"\b(to|by|at|of|in|on)[.!?](?:\s|$)", re.IGNORECASE
)
TO_SEQ_RE = re.compile(r"\bto \w+ to \w+\b", re.IGNORECASE)


def _log(pattern: str, match: Iterable[str]) -> None:
    logging.info("High-entropy pattern %s: %s", pattern, " ".join(match))


def passes_filters(text: str) -> bool:
    """Return True if ``text`` passes grammar filters."""

    if ARTICLE_PAIR_RE.search(text):
        return False
    if A_PREP_RE.search(text):
        return False
    for m in DUP_WORD_RE.finditer(text):
        word = m.group(1).lower()
        if word in DUP_WHITELIST:
            continue
        if word in VERB_SET:
            return False
        _log("duplicate", m.group(0).split())
    for m in SINGLE_LETTER_PAIR_RE.finditer(text):
        pair = m.group(0).lower()
        if pair in SINGLE_PAIR_WHITELIST:
            continue
        return False
    if SENTENCE_END_PREP_RE.search(text):
        _log("ending-preposition", SENTENCE_END_PREP_RE.search(text).group(0).split())
    if TO_SEQ_RE.search(text):
        _log("to-sequence", TO_SEQ_RE.search(text).group(0).split())
    return True
