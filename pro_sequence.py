from typing import Dict, List, Tuple


def analyze_sequences(state: Dict, words: List[str], char_n: int = 3) -> None:
    """Update state with word, bigram, trigram and char-level n-gram counts."""
    wc = state.setdefault('word_counts', {})
    bc = state.setdefault('bigram_counts', {})
    tc = state.setdefault('trigram_counts', {})
    cnc = state.setdefault('char_ngram_counts', {}) if char_n else None
    prev2 = '<s>'
    prev1 = '<s>'
    wc[prev1] = wc.get(prev1, 0) + 1
    wc[prev2] = wc.get(prev2, 0) + 1
    for word in words:
        wc[word] = wc.get(word, 0) + 1
        bc.setdefault(prev1, {})
        bc[prev1][word] = bc[prev1].get(word, 0) + 1
        key: Tuple[str, str] = (prev2, prev1)
        tc.setdefault(key, {})
        tc[key][word] = tc[key].get(word, 0) + 1
        if cnc is not None:
            for i in range(len(word) - char_n + 1):
                ngram = word[i : i + char_n]
                cnc[ngram] = cnc.get(ngram, 0) + 1
        prev2, prev1 = prev1, word
