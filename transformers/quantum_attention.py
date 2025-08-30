"""Experimental quantum-inspired attention mechanism.

This module simulates probability amplitudes and measurements to model a
quantum-style attention.  It is purely a toy implementation intended for
experiments and unit tests.
"""
from __future__ import annotations

import numpy as np


class QuantumAttention:
    """Simulate a minimal quantum attention mechanism.

    Parameters
    ----------
    noise : float, optional
        Standard deviation of complex gaussian noise applied to amplitudes.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, noise: float = 0.0, seed: int | None = None) -> None:
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def compute_amplitudes(self, query: np.ndarray, key: np.ndarray) -> np.ndarray:
        """Return normalised complex amplitudes for ``query`` and ``key``.

        The amplitudes are derived from the scaled dot product between query and
        key.  Complex gaussian noise can be injected to simulate decoherence.
        """

        scores = np.dot(query, key.T) / np.sqrt(key.shape[-1])
        amplitudes = np.exp(1j * scores)
        if self.noise:
            noise = self.noise * (
                self.rng.normal(size=amplitudes.shape)
                + 1j * self.rng.normal(size=amplitudes.shape)
            )
            amplitudes = amplitudes + noise
        # Normalise along the last axis
        norm = np.linalg.norm(amplitudes, axis=-1, keepdims=True)
        # Avoid division by zero
        norm = np.where(norm == 0, 1, norm)
        return amplitudes / norm

    def measure(self, amplitudes: np.ndarray, value: np.ndarray) -> np.ndarray:
        """Collapse ``amplitudes`` and return classical output.

        Probabilities are obtained by squaring the magnitude of the complex
        amplitudes.  These probabilities are used to take the expected value over
        ``value``.
        """

        probs = np.abs(amplitudes) ** 2
        probs = probs / probs.sum(axis=-1, keepdims=True)
        return probs @ value
