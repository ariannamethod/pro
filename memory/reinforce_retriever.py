"""Policy gradient memory retriever.

This module implements :class:`ReinforceRetriever`, a minimal reinforcement
learning policy that selects a memory node from ``MemoryStore`` using the
REINFORCE algorithm.  Memory texts are encoded into fixed-size vectors and a
softmax over a learnable weight vector produces a probability distribution over
nodes.  During retrieval the expected memory vector weighted by these
probabilities is returned which can be fed directly into cross-attention.  The
:py:meth:`update` method applies an online policy gradient step given a scalar
reward.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from morphology import encode as encode_morph
from .storage import MemoryStore


class ReinforceRetriever:
    """Select memory nodes using a lightweight policy gradient."""

    def __init__(self, store: MemoryStore, dim: int = 32, lr: float = 0.1) -> None:
        self.store = store
        self.dim = dim
        self.lr = lr
        # Parameters for a linear policy mapping encoded messages to logits
        self.w = np.zeros(dim, dtype=np.float32)
        # Cached state from the last retrieval for credit assignment
        self._last: Optional[Tuple[np.ndarray, np.ndarray, int]] = None

    # ------------------------------------------------------------------ utils
    def _encode(self, text: str) -> np.ndarray:
        """Encode ``text`` using morpheme hashing."""
        return encode_morph(text, self.dim)

    # ---------------------------------------------------------------- retrieval
    def retrieve(self, dialogue_id: str, speaker: str) -> np.ndarray:
        """Return a memory vector weighted by retrieval probabilities.

        The method also samples a concrete node whose index is stored so that
        :py:meth:`update` can assign credit based on an external reward.
        """

        dialogue = self.store.get_dialogue(dialogue_id)
        nodes = [n for n in dialogue if n.speaker == speaker]
        if not nodes:
            self._last = None
            return np.zeros(self.dim, dtype=np.float32)

        vecs = np.stack(
            [
                np.array(getattr(n, "morph_codes", None), dtype=np.float32)
                if getattr(n, "morph_codes", None) is not None
                else self._encode(n.text)
                for n in nodes
            ]
        )  # (n, dim)
        logits = vecs @ self.w  # (n,)
        # Numerical stability for softmax
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()
        idx = int(np.random.choice(len(nodes), p=probs))
        self._last = (vecs, probs, idx)
        # Expected memory vector used for attention
        weighted = probs @ vecs
        return weighted

    # ------------------------------------------------------------------ update
    def update(self, reward: float) -> None:
        """Apply an online REINFORCE update with ``reward``."""
        if self._last is None:
            return
        vecs, probs, idx = self._last
        baseline = probs @ vecs
        grad = vecs[idx] - baseline
        self.w += self.lr * reward * grad
        # Clear state so updates correspond to a single retrieval
        self._last = None
