"""Time folding transformer that looks ahead and updates backwards.

The :class:`TimeFoldTransformer` runs a provided step function forward a number
of steps, caching each intermediate prediction.  Amplitude vectors are stored in
a :class:`~quantum_memory.QuantumMemory` so that reverse-time gradient updates
can be computed later via :meth:`echo_backward`.
"""
from __future__ import annotations

from typing import Callable, List

import numpy as np

from quantum_memory import QuantumMemory


class TimeFoldTransformer:
    """Run forward for multiple future steps and support reverse updates."""

    def __init__(
        self,
        step_fn: Callable[[np.ndarray], np.ndarray],
        steps: int,
        memory: QuantumMemory | None = None,
    ) -> None:
        self.step_fn = step_fn
        self.steps = steps
        self.memory = memory or QuantumMemory()
        self.cache: List[np.ndarray] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run ``step_fn`` forward ``steps`` times storing predictions."""

        self.cache = []
        out = x
        for i in range(self.steps):
            out = self.step_fn(out)
            self.cache.append(out)
            self.memory.store(i, out.astype(np.complex128))
        return out

    def echo_backward(self, grad: np.ndarray) -> np.ndarray:
        """Propagate ``grad`` backwards using stored amplitudes."""

        for i in reversed(range(self.steps)):
            amps = self.memory.retrieve(i)
            if amps is not None:
                grad = grad * np.conj(amps).real
        return grad
