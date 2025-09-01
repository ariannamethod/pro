"""Utilities for running 4-bit quantised weights on CPU."""

from __future__ import annotations

import numpy as np
from quantum.quant4 import decompress_weights


class Quant4Linear:
    """Minimal linear layer using 4-bit packed weights."""

    def __init__(self, packed: np.ndarray, scale: float, shape: tuple[int, int], bias: np.ndarray | None = None) -> None:
        self.packed = packed
        self.scale = scale
        self.shape = shape
        self.bias = bias

    def __call__(self, x: np.ndarray) -> np.ndarray:
        weight = decompress_weights(self.packed, self.scale, self.shape)
        out = x @ weight
        if self.bias is not None:
            out = out + self.bias
        return out


def load_quant4(path: str) -> Quant4Linear:
    """Load ``Quant4Linear`` weights from an ``npz`` file."""
    data = np.load(path)
    packed = data["packed"]
    scale = float(data["scale"])
    shape = tuple(data["shape"])
    bias = data.get("bias")
    return Quant4Linear(packed, scale, shape, bias)
