import logging
import re
from typing import Iterable

# Strictly forbidden patterns ---------------------------------------------

ARTICLE_PAIR_RE = re.compile(r"\b(the|a|an) (the|a|an)\b", re.IGNORECASE)
# article 'a' before prepositions with no subsequent word
A_PREP_RE = re.compile(r"\ba (in|on|to|by|at|of)\b(?! \w)", re.IGNORECASE)
DUP_WORD_RE = re.compile(r"\b(\w+)\b \1\b", re.IGNORECASE)
SINGLE_LETTER_PAIR_RE = re.compile(r"\b\w\b \b\w\b")
MID_SENTENCE_CAP_THE_RE = re.compile(r"(?<!^)(?<![.!?]\s)The\b")
ARTICLE_PRONOUN_RE = re.compile(
    r"\b(the|a)\s+(he|she|they|it)\b", re.IGNORECASE
)

# Smart grammar patterns -------------------------------------------------
YOUR_YOU_RE = re.compile(r"\byour you\b", re.IGNORECASE)
I_TO_RE = re.compile(r"\bi to\b(?! \w*(ing|ed|go|see|be|do|have|get|come|want))", re.IGNORECASE)
I_MISSING_VERB_RE = re.compile(r"\bi (the|a|an|to|of|in|on|at|by|with|for)\b", re.IGNORECASE)

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


def _log(pattern: str, match) -> None:
    logging.info("High-entropy pattern %s: %s", pattern, " ".join(match))


def passes_filters(text: str) -> bool:
    """Return True if ``text`` passes grammar filters."""

    if ARTICLE_PAIR_RE.search(text):
        return False
    if A_PREP_RE.search(text):
        return False
    if MID_SENTENCE_CAP_THE_RE.search(text):
        return False
    if ARTICLE_PRONOUN_RE.search(text):
        return False
    # Новые умные правила
    if YOUR_YOU_RE.search(text):
        return False
    if I_TO_RE.search(text):
        return False
    if I_MISSING_VERB_RE.search(text):
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
    m = SENTENCE_END_PREP_RE.search(text)
    if m:
        token = m.group(0).rstrip(".!?")
        if not (token.islower() or token.isupper()):
            return False
        _log("ending-preposition", m.group(0).split())
    if TO_SEQ_RE.search(text):
        _log("to-sequence", TO_SEQ_RE.search(text).group(0).split())
    return True
