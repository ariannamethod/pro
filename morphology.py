"""Basic morphological analysis utilities.

This module provides a tiny helper for splitting a word into its root and
affixes.  The implementation is deliberately simple – it merely strips a set of
common prefixes and suffixes.  Results of the analysis are cached so repeated
calls for the same word are inexpensive.

The module exposes a single public function :func:`split` which returns a tuple
``(root, prefixes, suffixes)`` where prefixes and suffixes are returned as
lists preserving their order.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

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


__all__ = ["split"]

