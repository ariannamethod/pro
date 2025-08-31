"""Complex-valued self-attention using ``torch.view_as_complex``.

This is a tiny demonstrative implementation that interprets real and
imaginary parts stacked along the last dimension.  Inputs are expected to
have shape ``(..., 2 * dim)`` where the final dimension alternates real and
imaginary components.  The module converts them into complex tensors,
performs a scaled dot-product attention in the complex domain and returns the
result flattened back to real/imag parts.
"""

from __future__ import annotations

import math
import torch
from torch import nn


def _to_complex(x: torch.Tensor) -> torch.Tensor:
    """View the last dimension of ``x`` as complex numbers."""
    # ``torch.view_as_complex`` expects an extra dimension of size 2 holding
    # the real and imaginary parts respectively.
    x = x.view(*x.shape[:-1], -1, 2)
    return torch.view_as_complex(x)


class ComplexAttention(nn.Module):
    """Minimal complex self-attention layer."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        # projections produce real and imaginary parts for Q, K, V
        self.q_proj = nn.Linear(dim, 2 * dim)
        self.k_proj = nn.Linear(dim, 2 * dim)
        self.v_proj = nn.Linear(dim, 2 * dim)
        self.out_proj = nn.Linear(2 * dim, dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return attention output for ``hidden_states``.

        ``hidden_states`` is interpreted as real-valued input.  It is projected
        to complex queries, keys and values.  Attention weights are computed
        from the real part of the complex scores.  The complex result is
        flattened back to real/imag parts before a final linear projection.
        """

        q = _to_complex(self.q_proj(hidden_states))
        k = _to_complex(self.k_proj(hidden_states))
        v = _to_complex(self.v_proj(hidden_states))

        scale = 1.0 / math.sqrt(self.dim)
        scores = torch.matmul(q, k.conj().transpose(-2, -1)) * scale
        weights = torch.softmax(scores.real, dim=-1)
        out = torch.matmul(weights, v)
        out = torch.view_as_real(out).reshape(*hidden_states.shape[:-1], -1)
        return self.out_proj(out)
