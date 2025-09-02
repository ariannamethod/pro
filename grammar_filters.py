"""Грамматические фильтры для SmallTalk PRO"""

import re

# Базовые правила (совместимые с Python 3.7)
ARTICLE_PAIR_RE = re.compile(r"\b(a|an)\s+(a|an)\b", re.IGNORECASE)
A_PREP_RE = re.compile(r"\ba\s+(of|in|on|at|by|with|for)\b", re.IGNORECASE)
DUP_WORD_RE = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)
SINGLE_LETTER_PAIR_RE = re.compile(r"\b[a-zA-Z]\s+[a-zA-Z]\b")
MID_SENTENCE_CAP_THE_RE = re.compile(r"\bThe\b")
ARTICLE_PRONOUN_RE = re.compile(r"\b(a|an)\s+(I|you|he|she|it|we|they)\b", re.IGNORECASE)

# НОВЫЕ ПРАВИЛА ПРОТИВ ПОВТОРОВ И ИНВЕРСИИ
YOUR_YOU_RE = re.compile(r"\byour you\b", re.IGNORECASE)
I_TO_RE = re.compile(r"\bi to\b(?! \w*(ing|ed|go|see|be|do|have|get|come|want))", re.IGNORECASE)
I_MISSING_VERB_RE = re.compile(r"\bi (the|a|an|to|of|in|on|at|by|with|for)\b", re.IGNORECASE)
PRONOUN_CLUSTER_RE = re.compile(r"\b(i|you|he|she|it|we|they)\s+(i|you|he|she|it|we|they)\b", re.IGNORECASE)
PRONOUN_THE_RE = re.compile(r"\b(i|you|he|she|it|we|they)\s+the\s+(i|you|he|she|it|we|they)\b", re.IGNORECASE)
RECURSIVE_PATTERN_RE = re.compile(r"\b(\w+\.)\s+\1\b", re.IGNORECASE)
TRIPLE_WORD_RE = re.compile(r"\b(\w+)\s+\1\s+\1\b", re.IGNORECASE)
S_TOKEN_RE = re.compile(r"\b<s>\b", re.IGNORECASE)
SINGLE_LETTER_CYCLE_RE = re.compile(r"\b(\w)\.\s+(\w)\.\s+\1\.\s+\2\.\b", re.IGNORECASE)
WORD_REPEAT_RE = re.compile(r"\b(you|the|a|an|i|we|they|he|she|it)\s+\1\b", re.IGNORECASE)

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
    if SINGLE_LETTER_PAIR_RE.search(text):
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
