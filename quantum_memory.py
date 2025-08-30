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
    """Store complex amplitudes for discrete time steps."""

    def __init__(self) -> None:
        self.amplitudes: Dict[int, np.ndarray] = {}

    def store(self, step: int, amps: np.ndarray) -> None:
        """Store ``amps`` for ``step``."""

        self.amplitudes[step] = amps

    def retrieve(self, step: int) -> np.ndarray | None:
        """Return stored amplitudes for ``step`` if present."""

        return self.amplitudes.get(step)

    def clear(self) -> None:
        """Remove all stored amplitudes."""

        self.amplitudes.clear()
