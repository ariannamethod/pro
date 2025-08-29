from typing import Dict, List

def analyze_sequences(state: Dict, words: List[str]) -> None:
    """Update state with word and bigram counts."""
    wc = state.setdefault('word_counts', {})
    bc = state.setdefault('bigram_counts', {})
    prev = '<s>'
    wc[prev] = wc.get(prev, 0) + 1
    for word in words:
        wc[word] = wc.get(word, 0) + 1
        bc.setdefault(prev, {})
        bc[prev][word] = bc[prev].get(word, 0) + 1
        prev = word
