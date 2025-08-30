"""Fractal resonance adapter with recursive block construction."""

from __future__ import annotations

import numpy as np


class FractalAdapter:
    """Generate a recursive sinusoidal pattern.

    The adapter builds a fractal-like vector by recursively splitting the
    dimension and applying scaled sine waves at each depth.  The result can be
    added to hidden states to inject a deterministic but rich signal.
    """

    def __init__(
        self,
        depth: int,
        frequency: float = 1.0,
        amplitude: float = 0.1,
    ) -> None:
        self.depth = max(depth, 0)
        self.frequency = frequency
        self.amplitude = amplitude

    def _block(
        self, dim: int, depth: int, freq: float, amp: float
    ) -> np.ndarray:
        base = amp * np.sin(freq * np.arange(dim, dtype=np.float32))
        if depth == 0 or dim <= 1:
            return base
        half = dim // 2
        left = self._block(half, depth - 1, freq * 2.0, amp / 2.0)
        right = self._block(dim - half, depth - 1, freq * 2.0, amp / 2.0)
        return base + np.concatenate([left, right])

    def __call__(self, dim: int) -> np.ndarray:
        """Return an adapter vector of length ``dim``."""

        return self._block(dim, self.depth, self.frequency, self.amplitude)
