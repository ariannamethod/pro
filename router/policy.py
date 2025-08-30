from __future__ import annotations

"""Routing policy for hybrid quantum/classical attention."""

import numpy as np


class PatchRoutingPolicy:
    """Simple REINFORCE-style policy for routing patches.

    The policy is represented by a weight vector that maps patch features to a
    probability of using the quantum backend.  After each decision a reward can
    be supplied to ``update`` which adjusts the weights via the REINFORCE
    gradient estimator.
    """

    def __init__(self, dim: int, lr: float = 0.1, seed: int | None = None) -> None:
        self.weights = np.zeros(dim, dtype=float)
        self.lr = lr
        self.rng = np.random.default_rng(seed)
        self._last: tuple[np.ndarray, bool, float] | None = None

    def route(self, features: np.ndarray) -> np.ndarray:
        """Return a boolean mask indicating quantum routing decisions."""
        logits = features @ self.weights
        probs = 1 / (1 + np.exp(-logits))
        decisions = self.rng.random(size=probs.shape) < probs
        self._last = (features, decisions, probs)
        return decisions

    def update(self, reward: float) -> None:
        """Update policy weights given ``reward``."""
        if self._last is None:
            return
        features, decisions, probs = self._last
        grad = (decisions.astype(float) - probs)[:, None] * features
        self.weights += self.lr * reward * grad.mean(axis=0)
        self._last = None
