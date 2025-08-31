"""Simplified transformer block that can use complex attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - handled in tests
    torch = None
    nn = object  # type: ignore[misc,assignment]

from attention import ComplexAttention


@dataclass
class TransformerBlockConfig:
    """Configuration for :class:`TransformerBlock`.

    Parameters
    ----------
    dim: int
        Dimensionality of the input embeddings.
    use_complex_attention: bool, optional
        Whether to enable :class:`ComplexAttention` inside the block.
    """

    dim: int
    use_complex_attention: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation of this config."""
        return {"dim": self.dim, "use_complex_attention": self.use_complex_attention}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformerBlockConfig":
        """Instantiate config from ``data``.

        Missing ``use_complex_attention`` defaults to ``False`` for backwards
        compatibility.
        """
        return cls(dim=int(data["dim"]), use_complex_attention=bool(data.get("use_complex_attention", False)))


class TransformerBlock(nn.Module if torch else object):  # type: ignore[misc]
    """Very small transformer block wrapper."""

    def __init__(self, config: TransformerBlockConfig) -> None:  # pragma: no cover - simple wiring
        if torch is None:
            raise ModuleNotFoundError("torch is required for TransformerBlock")
        super().__init__()
        self.config = config
        self.attention = (
            ComplexAttention(config.dim) if config.use_complex_attention else None
        )

    def forward(self, hidden_states: "torch.Tensor") -> "torch.Tensor":  # pragma: no cover - thin wrapper
        if self.attention is not None:
            return self.attention(hidden_states)
        return hidden_states

    __call__ = forward
