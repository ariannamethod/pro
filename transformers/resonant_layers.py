"""Simple harmonic layers for transformer experiments."""

from __future__ import annotations

import numpy as np


class HarmonicResonanceLayer:
    """Apply a sinusoidal basis with shared weights.

    Parameters
    ----------
    dim: int
        Dimensionality of the hidden states.
    frequencies: Sequence[float]
        Frequencies used to construct the Fourier basis.  All instances of
        :class:`HarmonicResonanceLayer` share the same weights which are
        initialised on first construction.
    """

    _shared_weights: np.ndarray | None = None

    def __init__(self, dim: int, frequencies: np.ndarray) -> None:
        self.dim = dim
        self.frequencies = np.asarray(frequencies, dtype=np.float32)
        if (
            HarmonicResonanceLayer._shared_weights is None
            or HarmonicResonanceLayer._shared_weights.shape[0] != len(self.frequencies)
        ):
            HarmonicResonanceLayer._shared_weights = np.ones(
                len(self.frequencies), dtype=np.float32
            )

    @property
    def weights(self) -> np.ndarray:
        assert HarmonicResonanceLayer._shared_weights is not None
        return HarmonicResonanceLayer._shared_weights

    def modulate(self, memory: np.ndarray) -> None:
        """Blend *memory* into the shared weights."""
        w = self.weights
        HarmonicResonanceLayer._shared_weights = w + memory[: w.shape[0]]

    def __call__(self, hidden_states: np.ndarray) -> np.ndarray:
        """Compute the harmonic component for ``hidden_states``."""
        idx = np.arange(self.dim, dtype=np.float32)
        basis = np.sin(np.outer(idx, self.frequencies))
        return basis @ self.weights
