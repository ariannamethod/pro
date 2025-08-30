from typing import Dict, List, Tuple, Optional


def analyze_sequences(
    state: Dict,
    words: List[str],
    char_n: int = 3,
    weight: float = 1.0,
    window_size: Optional[int] = None,
) -> None:
    """Update state with n-gram counts within a recent context window.

    Only the last *window_size* tokens of *words* are considered when
    updating counts. If *window_size* is ``None`` all tokens are used.
    """
    if window_size:
        words = words[-window_size:]
    wc = state.setdefault('word_counts', {})
    bc = state.setdefault('bigram_counts', {})
    tc = state.setdefault('trigram_counts', {})
    cnc = state.setdefault('char_ngram_counts', {}) if char_n else None
    # Inverse-frequency maps
    wi = state.setdefault('word_inv', {})
    bi = state.setdefault('bigram_inv', {})
    ti = state.setdefault('trigram_inv', {})
    cni = state.setdefault('char_ngram_inv', {}) if char_n else None

    prev2 = '<s>'
    prev1 = '<s>'
    wc[prev1] = wc.get(prev1, 0) + weight
    wi[prev1] = 1.0 / wc[prev1]
    wc[prev2] = wc.get(prev2, 0) + weight
    wi[prev2] = 1.0 / wc[prev2]
    for word in words:
        wc[word] = wc.get(word, 0) + weight
        wi[word] = 1.0 / wc[word]
        bc.setdefault(prev1, {})
        bc[prev1][word] = bc[prev1].get(word, 0) + weight
        bi.setdefault(prev1, {})
        bi[prev1][word] = 1.0 / bc[prev1][word]
        key: Tuple[str, str] = (prev2, prev1)
        tc.setdefault(key, {})
        tc[key][word] = tc[key].get(word, 0) + weight
        ti.setdefault(key, {})
        ti[key][word] = 1.0 / tc[key][word]
        if cnc is not None:
            for i in range(len(word) - char_n + 1):
                ngram = word[i:i + char_n]
                cnc[ngram] = cnc.get(ngram, 0) + weight
                cni[ngram] = 1.0 / cnc[ngram]
        prev2, prev1 = prev1, word
