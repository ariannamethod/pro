"""Basic morphological analysis utilities.

This module provides helpers for splitting a word into morphemes and for
producing a simple *resonant* encoding of a text.  The implementation is
deliberately small – it merely strips a set of common prefixes and suffixes and
hashes resulting morphemes into a fixed-size numeric vector.  Results of the
analysis are cached so repeated calls for the same word are inexpensive.

The main public functions are:

``split``
    Split a single word into ``(root, prefixes, suffixes)``.
``tokenize``
    Break a text into a flat list of morphemes.
``encode``
    Aggregate morphemes from a text into a deterministic fixed-size vector.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import re
from typing import List, Tuple

import numpy as np

# A tiny list of common Russian prefixes and suffixes.  The lists are not
# exhaustive; they merely provide enough variety for simple experiments.
_PREFIXES = [
    "без",
    "под",
    "над",
    "про",
    "пред",
    "раз",
    "по",
    "на",
]

_SUFFIXES = [
    "ение",
    "ость",
    "ировать",
    "ить",
    "ать",
    "овать",
    "ка",
    "ник",
    "ский",
    "ый",
    "ой",
    "а",
    "я",
    "ы",
    "и",
]


@lru_cache(maxsize=2048)
def split(word: str) -> Tuple[str, List[str], List[str]]:
    """Split *word* into a root and its affixes.

    Parameters
    ----------
    word:
        The word to analyse.

    Returns
    -------
    tuple
        A tuple ``(root, prefixes, suffixes)``.  The ``prefixes`` and
        ``suffixes`` lists preserve the order of affixes as they appear in the
        original word.
    """

    prefixes: List[str] = []
    suffixes: List[str] = []
    root = word

    # Strip prefixes greedily from longest to shortest to avoid ambiguous
    # splits such as "под" + "над" + root.
    changed = True
    while changed:
        changed = False
        for pref in sorted(_PREFIXES, key=len, reverse=True):
            if root.startswith(pref) and len(root) > len(pref) + 1:
                prefixes.append(pref)
                root = root[len(pref) :]
                changed = True
                break

    # Similarly strip suffixes greedily.
    changed = True
    while changed:
        changed = False
        for suff in sorted(_SUFFIXES, key=len, reverse=True):
            if root.endswith(suff) and len(root) > len(suff) + 1:
                suffixes.append(suff)
                root = root[: -len(suff)]
                changed = True
                break

    return root, prefixes, suffixes[::-1]


def tokenize(text: str) -> List[str]:
    """Return a flat list of morphemes extracted from ``text``.

    The function splits ``text`` into words, applies :func:`split` to each and
    flattens the resulting prefixes, root and suffixes into a single list.  All
    words are lowercased and non-word characters are ignored.
    """

    morphs: List[str] = []
    for word in re.findall(r"\w+", text.lower()):
        root, prefixes, suffixes = split(word)
        morphs.extend(prefixes + [root] + suffixes)
    return morphs


def encode(text: str, dim: int = 32) -> np.ndarray:
    """Encode ``text`` into a fixed-size vector using morpheme hashing.

    Parameters
    ----------
    text:
        Input text to encode.
    dim:
        Dimension of the resulting vector.
    """

    vec = np.zeros(dim, dtype=np.float32)
    for morph in tokenize(text):
        h = hashlib.md5(morph.encode("utf-8")).hexdigest()
        idx = int(h, 16) % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


__all__ = ["split", "tokenize", "encode"]

