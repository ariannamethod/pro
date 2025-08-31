"""Complex-valued self-attention implemented with NumPy.

This is a tiny demonstrative implementation that interprets real and
imaginary parts stacked along the last dimension. Inputs are expected to
have shape ``(..., 2 * dim)`` where the final dimension alternates real and
imaginary components. The module converts them into complex arrays,
performs a scaled dot-product attention in the complex domain and returns
the result flattened back to real/imag parts.
"""

from __future__ import annotations

import math
import numpy as np


def _to_complex(x: np.ndarray) -> np.ndarray:
    """View the last dimension of ``x`` as complex numbers."""
    x = x.reshape(*x.shape[:-1], -1, 2)
    return x[..., 0] + 1j * x[..., 1]


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return the softmax of ``x`` along ``axis``."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class Linear:
    """Simple linear projection implemented with NumPy arrays."""

    def __init__(self, in_features: int, out_features: int) -> None:
        rng = np.random.default_rng()
        self.weight = rng.standard_normal((out_features, in_features))
        self.bias = np.zeros(out_features)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weight.T + self.bias


class ComplexAttention:
    """Minimal complex self-attention layer using NumPy arrays."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.q_proj = Linear(dim, 2 * dim)
        self.k_proj = Linear(dim, 2 * dim)
        self.v_proj = Linear(dim, 2 * dim)
        self.out_proj = Linear(2 * dim, dim)

    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """Return attention output for ``hidden_states``.

        ``hidden_states`` is interpreted as real-valued input. It is projected
        to complex queries, keys and values. Attention weights are computed
        from the real part of the complex scores. The complex result is
        flattened back to real/imag parts before a final linear projection.
        """

        q = _to_complex(self.q_proj(hidden_states))
        k = _to_complex(self.k_proj(hidden_states))
        v = _to_complex(self.v_proj(hidden_states))

        scale = 1.0 / math.sqrt(self.dim)
        scores = np.matmul(q, np.conjugate(np.swapaxes(k, -2, -1))) * scale
        weights = _softmax(scores.real, axis=-1)
        out = np.matmul(weights, v)
        out = np.stack((out.real, out.imag), axis=-1).reshape(
            *hidden_states.shape[:-1], -1
        )
        return self.out_proj(out)

    __call__ = forward
