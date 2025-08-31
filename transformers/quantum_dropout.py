"""Random phase rotation followed by magnitude projection."""
from __future__ import annotations

import numpy as np


def quantum_dropout(x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return ``x`` after random phase rotation and measurement.

    Each element of ``x`` is treated as the magnitude of a complex amplitude.
    A random phase is sampled, the amplitude is rotated by this phase and the
    absolute value of its real projection is returned. The procedure mimics a
    probabilistic ``dropout`` where components can vanish depending on the
    sampled phase.

    Parameters
    ----------
    x:
        Real input array representing amplitudes.
    rng:
        Optional random number generator. If ``None`` a fresh generator is
        created, leading to non-deterministic behaviour even when global seeds
        are fixed.
    """

    if rng is None:
        rng = np.random.default_rng()
    phase = rng.uniform(0.0, 2 * np.pi, size=x.shape)
    rotated = x.astype(np.complex128) * np.exp(1j * phase)
    return np.abs(rotated.real)
