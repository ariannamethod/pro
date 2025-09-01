"""Amplitude-based memory for storing future states.

This module provides :class:`QuantumMemory`, a minimal container that keeps
complex amplitude vectors for a sequence of future time steps.  The memory is
indexed by step number and can be retrieved later when performing reverse-time
updates.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


class QuantumMemory:
    """Store complex amplitudes for discrete time steps.

    Tokens can be provided directly.  They are converted into a fractal
    representation whose harmonics embody semantic "resonances" at multiple
    scales.  The encoding is intentionally light-weight – it does not attempt
    to be linguistically perfect, only to provide a deterministic mapping from
    a token to a complex vector whose phases repeat self‑similar patterns.
    """

    def __init__(self) -> None:
        self.amplitudes: Dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Fractal token encoding
    def fractal_encode(self, token: str, depth: int = 4) -> np.ndarray:
        """Return a fractal embedding for ``token``.

        Each level halves the driving frequency which produces a set of
        harmonics reminiscent of a Mandelbrot zoom.  Summing character codes
        yields a reproducible seed so the same token always maps to the same
        spiral of complex amplitudes.
        """

        seed = sum(ord(c) for c in token)
        amps = [np.exp(1j * seed / (2 ** i)) for i in range(1, depth + 1)]
        return np.asarray(amps, dtype=np.complex64)

    def semantic_resonance(self, a: str, b: str) -> float:
        """Measure resonance between two tokens.

        Cosine similarity in the fractal space acts as a lightweight notion of
        semantic overlap.
        """

        va = self.fractal_encode(a)
        vb = self.fractal_encode(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 0.0
        return float(np.real(np.vdot(va, vb)) / denom)

    # ------------------------------------------------------------------
    def store(self, step: int, data: str | np.ndarray) -> None:
        """Store ``data`` for ``step``.

        ``data`` may be a complex amplitude array or a raw token.  When a
        token is supplied it is converted to its fractal resonance embedding
        first.
        """

        if isinstance(data, np.ndarray):
            amps = data
        else:
            amps = self.fractal_encode(data)
        self.amplitudes[step] = amps

    def retrieve(self, step: int) -> np.ndarray | None:
        """Return stored amplitudes for ``step`` if present."""

        return self.amplitudes.get(step)

    def clear(self) -> None:
        """Remove all stored amplitudes."""

        self.amplitudes.clear()
