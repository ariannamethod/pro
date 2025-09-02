"""Грамматические фильтры для SmallTalk PRO"""

import re
from typing import Optional

# Базовые правила (совместимые с Python 3.7)
ARTICLE_PAIR_RE = re.compile(r"\b(a|an)\s+(a|an)\b", re.IGNORECASE)
A_PREP_RE = re.compile(r"\ba\s+(of|in|on|at|by|with|for)\b", re.IGNORECASE)
DUP_WORD_RE = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)
SINGLE_LETTER_PAIR_RE = re.compile(r"\b[a-zA-Z]\s+[a-zA-Z]\b")
MID_SENTENCE_CAP_THE_RE = re.compile(r"(?<!^)(?<![.!?…]\s)\bThe\b", re.MULTILINE)
ARTICLE_PRONOUN_RE = re.compile(r"\b(a|an)\s+(I|you|he|she|it|we|they)\b", re.IGNORECASE)

# НОВЫЕ ПРАВИЛА ПРОТИВ ПОВТОРОВ И ИНВЕРСИИ
YOUR_YOU_RE = re.compile(r"\byour you\b", re.IGNORECASE)
I_TO_RE = re.compile(r"\bi to\b(?! \w*(ing|ed|go|see|be|do|have|get|come|want))", re.IGNORECASE)
I_MISSING_VERB_RE = re.compile(r"\bi (the|a|an|to|of|in|on|at|by|with|for)\b", re.IGNORECASE)

# Expanded pronouns (subject/object/possessive/reflexive)
PRONOUNS = r"(?:i|you|he|she|it|we|they|me|him|her|us|them|my|your|his|its|our|their|mine|yours|hers|ours|theirs|myself|yourself|himself|herself|itself|ourselves|yourselves|themselves)"
PRONOUN_CLUSTER_RE = re.compile(rf"\b{PRONOUNS}\s+{PRONOUNS}\b", re.IGNORECASE)
PRONOUN_THE_RE = re.compile(r"\b(i|you|he|she|it|we|they)\s+the\s+(i|you|he|she|it|we|they)\b", re.IGNORECASE)
RECURSIVE_PATTERN_RE = re.compile(r"\b(\w+\.)\s+\1\b", re.IGNORECASE)
TRIPLE_WORD_RE = re.compile(r"\b(\w+)\s+\1\s+\1\b", re.IGNORECASE)
S_TOKEN_RE = re.compile(r"\b<s>\b", re.IGNORECASE)
SINGLE_LETTER_CYCLE_RE = re.compile(r"\b(\w)\.\s+(\w)\.\s+\1\.\s+\2\.\b", re.IGNORECASE)
WORD_REPEAT_RE = re.compile(r"\b(you|the|a|an|i|we|they|he|she|it)\s+\1\b", re.IGNORECASE)

def _has_single_letter_streak(text: str, max_run: int = 2) -> bool:
    """Check if text has more than max_run single-letter words in a row."""
    words = re.findall(r"\b\w+\b", text)
    run = 0
    for w in words:
        if len(w) == 1 and w.isalpha():
            run += 1
            if run > max_run:
                return True
        else:
            run = 0
    return False

def passes_filters(text: str) -> bool:
    """Проверяет текст на соответствие грамматическим правилам"""
    if not text or len(text.strip()) == 0:
        return False
    
    # Базовые правила
    if ARTICLE_PAIR_RE.search(text):
        return False
    if A_PREP_RE.search(text):
        return False
    if DUP_WORD_RE.search(text):
        return False
    if _has_single_letter_streak(text, max_run=2):
        return False
    if MID_SENTENCE_CAP_THE_RE.search(text):
        return False
    if ARTICLE_PRONOUN_RE.search(text):
        return False
    
    # НОВЫЕ ПРАВИЛА
    if YOUR_YOU_RE.search(text):
        return False
    if I_TO_RE.search(text):
        return False
    if I_MISSING_VERB_RE.search(text):
        return False
    if PRONOUN_CLUSTER_RE.search(text):
        return False
    if PRONOUN_THE_RE.search(text):
        return False
    if RECURSIVE_PATTERN_RE.search(text):
        return False
    if TRIPLE_WORD_RE.search(text):
        return False
    if S_TOKEN_RE.search(text):
        return False
    if SINGLE_LETTER_CYCLE_RE.search(text):
        return False
    if WORD_REPEAT_RE.search(text):
        return False
    
    return True

def swap_pronouns(words):
    """Меняет местоимения местами"""
    pronoun_map = {
        "I": "you", "i": "you",
        "you": "I", "You": "I", 
        "my": "your", "My": "Your",
        "your": "my", "Your": "My",
        "me": "you", "Me": "You",
        "myself": "yourself", "yourself": "myself",
        "mine": "yours", "yours": "mine"
    }
    
    return [pronoun_map.get(word, word) for word in words]

from typing import Optional

def debug_filters(text: str) -> tuple[bool, Optional[str]]:
    """Debug helper that returns whether text passes and the first rule name that failed."""
    if not text or len(text.strip()) == 0:
        return False, "empty_text"
    
    # Базовые правила
    if ARTICLE_PAIR_RE.search(text):
        return False, "ARTICLE_PAIR_RE"
    if A_PREP_RE.search(text):
        return False, "A_PREP_RE"
    if DUP_WORD_RE.search(text):
        return False, "DUP_WORD_RE"
    if _has_single_letter_streak(text, max_run=2):
        return False, "single_letter_streak"
    if MID_SENTENCE_CAP_THE_RE.search(text):
        return False, "MID_SENTENCE_CAP_THE_RE"
    if ARTICLE_PRONOUN_RE.search(text):
        return False, "ARTICLE_PRONOUN_RE"
    
    # НОВЫЕ ПРАВИЛА
    if YOUR_YOU_RE.search(text):
        return False, "YOUR_YOU_RE"
    if I_TO_RE.search(text):
        return False, "I_TO_RE"
    if I_MISSING_VERB_RE.search(text):
        return False, "I_MISSING_VERB_RE"
    if PRONOUN_CLUSTER_RE.search(text):
        return False, "PRONOUN_CLUSTER_RE"
    if PRONOUN_THE_RE.search(text):
        return False, "PRONOUN_THE_RE"
    if RECURSIVE_PATTERN_RE.search(text):
        return False, "RECURSIVE_PATTERN_RE"
    if TRIPLE_WORD_RE.search(text):
        return False, "TRIPLE_WORD_RE"
    if S_TOKEN_RE.search(text):
        return False, "S_TOKEN_RE"
    if SINGLE_LETTER_CYCLE_RE.search(text):
        return False, "SINGLE_LETTER_CYCLE_RE"
    if WORD_REPEAT_RE.search(text):
        return False, "WORD_REPEAT_RE"
    
    return True, None
