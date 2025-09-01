import numpy as np
from typing import Tuple


def compress_weights(weights: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, ...]]:
    """Compress a float32 weight matrix to 4-bit representation.

    Returns
    -------
    packed : np.ndarray
        Array of type ``uint8`` packing two signed 4-bit values per byte.
    scale : float
        Scaling factor used during quantisation.
    shape : tuple[int, ...]
        Original shape of *weights* for later reconstruction.
    """
    if weights.dtype != np.float32:
        weights = weights.astype(np.float32)
    max_abs = float(np.max(np.abs(weights)))
    if max_abs == 0.0:
        max_abs = 1.0
    scale = max_abs / 7.0
    q = np.clip(np.round(weights / scale), -8, 7).astype(np.int8).ravel()
    if q.size % 2:
        q = np.pad(q, (0, 1), mode="constant")
    packed = ((q[0::2] & 0xF) << 4) | (q[1::2] & 0xF)
    return packed.astype(np.uint8), scale, weights.shape


def decompress_weights(packed: np.ndarray, scale: float, shape: Tuple[int, ...]) -> np.ndarray:
    """Restore 4-bit packed weights back to float32 array."""
    q = np.empty(packed.size * 2, dtype=np.int8)
    q[0::2] = (packed >> 4) & 0xF
    q[1::2] = packed & 0xF
    q = ((q + 8) % 16) - 8
    q = q[: np.prod(shape)]
    return (q.astype(np.float32) * scale).reshape(shape)
