"""Experimental quantum-inspired attention mechanism.

This module simulates probability amplitudes and measurements to model a
quantum-style attention.  It is purely a toy implementation intended for
experiments and unit tests.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - only used for typing
    from quantum.attention_backend import AttentionBackend  # noqa: F401


def _count_components(mask: np.ndarray) -> int:
    """Return number of 4-connected ``True`` components in *mask*."""
    visited = np.zeros_like(mask, dtype=bool)
    count = 0
    stack = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] and not visited[i, j]:
                count += 1
                stack = [(i, j)]
                visited[i, j] = True
                while stack:
                    x, y = stack.pop()
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < mask.shape[0] and 0 <= ny < mask.shape[1]:
                            if mask[nx, ny] and not visited[nx, ny]:
                                visited[nx, ny] = True
                                stack.append((nx, ny))
    return count


def _betti_features(amplitudes: np.ndarray) -> np.ndarray:
    """Return a simple Betti-0/1 estimate from amplitude magnitudes."""
    mags = np.abs(amplitudes)
    thresh = mags.mean()
    mask = mags > thresh
    if mask.ndim == 1:
        mask = mask[None, :]
    betti0 = _count_components(mask)
    padded = np.pad(~mask, 1, constant_values=True)
    holes = _count_components(padded) - 1
    betti1 = max(0, holes)
    return np.array([betti0, betti1], dtype=np.int64)


class QuantumAttention:
    """Simulate a minimal quantum attention mechanism.

    Implements :class:`~quantum.attention_backend.AttentionBackend` using a
    pure NumPy simulation.

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

    def compute_amplitudes(
        self, query: np.ndarray, key: np.ndarray
    ) -> np.ndarray:
        """Return normalised complex amplitudes for ``query`` and ``key``.

        The amplitudes are derived from the scaled dot product between query
        and key.  Complex gaussian noise can be injected to simulate
        decoherence.
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

    def measure(
        self, amplitudes: np.ndarray, value: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Collapse ``amplitudes`` and return output and Betti features.

        Probabilities are obtained by squaring the magnitude of the complex
        amplitudes. These probabilities are used to take the expected value
        over ``value``.  The magnitude grid is thresholded at its mean to
        compute Betti-0/1 estimates for connected components and holes.
        """

        probs = np.abs(amplitudes) ** 2
        probs = probs / probs.sum(axis=-1, keepdims=True)
        betti = _betti_features(amplitudes)
        return probs @ value, betti

    def attention(
        self, query: np.ndarray, key: np.ndarray, value: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convenience wrapper computing amplitudes then measuring them."""

        amplitudes = self.compute_amplitudes(query, key)
        return self.measure(amplitudes, value)
