"""Experimental semantic collider integrating morphological encoding, quantum memory and peer resonance."""
from __future__ import annotations

import itertools
from typing import Dict, List

import numpy as np

from morphology import encode
from quantum_memory import QuantumMemory
from resonance.p2p_resonance import P2PResonance


class SemanticCollider:
    """Combine morphology, quantum memory and P2P resonance.

    The collider operates in three steps:
    1. generate morpheme superpositions for candidate texts;
    2. measure pairwise semantic distances between them;
    3. compress the closest pair into a phrase.

    Additionally, ``scores`` can be used to obtain next-word scores for a
    vocabulary given a context.
    """

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim
        self.memory = QuantumMemory()
        self.resonance = P2PResonance()
        self._step = 0

    # ------------------------------------------------------------------
    def _superpositions(self, texts: List[str]) -> List[np.ndarray]:
        """Encode ``texts`` into morpheme-based vectors."""
        return [encode(t, dim=self.dim) for t in texts]

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        """Return a simple cosine distance between vectors."""
        if a.size == 0 or b.size == 0:
            return 1.0
        return float(1.0 - float(a @ b))

    # ------------------------------------------------------------------
    def collide(self, texts: List[str]) -> str:
        """Run the full collider cycle on ``texts`` and return a phrase."""
        if not texts:
            return ""

        vecs = self._superpositions(texts)
        for vec in vecs:
            self.memory.store(self._step, vec)
            self._step += 1

        best_pair: tuple[str, str] | None = None
        best_dist = float("inf")
        for (i, vi), (j, vj) in itertools.combinations(enumerate(vecs), 2):
            dist = self._distance(vi, vj)
            self.resonance.queue_update({f"d_{i}_{j}": -dist})
            if dist < best_dist:
                best_dist = dist
                best_pair = (texts[i], texts[j])

        if best_pair is None:
            return texts[0]
        return f"{best_pair[0]} {best_pair[1]}"

    # ------------------------------------------------------------------
    def scores(self, context: List[str], vocab: List[str]) -> Dict[str, float]:
        """Return scores for ``vocab`` words based on ``context`` tokens."""
        ctx_vec = encode(" ".join(context), dim=self.dim)
        scores: Dict[str, float] = {}
        for word in vocab:
            cand_vec = encode(" ".join(context + [word]), dim=self.dim)
            dist = self._distance(ctx_vec, cand_vec)
            self.memory.store(self._step, cand_vec)
            self._step += 1
            self.resonance.queue_update({word: -dist})
            scores[word] = -dist
        return scores


__all__ = ["SemanticCollider"]
