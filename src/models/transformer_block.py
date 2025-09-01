"""Simplified transformer block implemented with NumPy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from attention import ComplexAttention


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return the softmax of ``x`` along ``axis``."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class Linear:
    """Minimal linear layer backed by NumPy arrays."""

    def __init__(self, in_features: int, out_features: int) -> None:
        rng = np.random.default_rng()
        self.weight = rng.standard_normal((out_features, in_features))
        self.bias = np.zeros(out_features)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weight.T + self.bias

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"weight": self.weight, "bias": self.bias}

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        self.weight = state["weight"]
        self.bias = state["bias"]


@dataclass
class TransformerBlockConfig:
    """Configuration for :class:`TransformerBlock`."""

    dim: int
    use_complex_attention: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation of this config."""
        return {"dim": self.dim, "use_complex_attention": self.use_complex_attention}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformerBlockConfig":
        """Instantiate config from ``data``."""
        return cls(dim=int(data["dim"]), use_complex_attention=bool(data.get("use_complex_attention", False)))


class TransformerBlock:
    """Very small transformer block using NumPy operations."""

    def __init__(self, config: TransformerBlockConfig) -> None:
        self.config = config
        dim = config.dim
        if config.use_complex_attention:
            self.attention: ComplexAttention | None = ComplexAttention(dim)
            self.q_proj = self.k_proj = self.v_proj = self.out_proj = None
        else:
            self.attention = None
            self.q_proj = Linear(dim, dim)
            self.k_proj = Linear(dim, dim)
            self.v_proj = Linear(dim, dim)
            self.out_proj = Linear(dim, dim)

    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        if self.attention is not None:
            lattice = hidden_states.mean(axis=-1)
            return self.attention(hidden_states, resistance=lattice)
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        scores = np.matmul(q, np.swapaxes(k, -2, -1)) / math.sqrt(self.config.dim)
        weights = _softmax(scores, axis=-1)
        out = np.matmul(weights, v)
        return self.out_proj(out)

    __call__ = forward

    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {"config": self.config.to_dict()}
        if self.attention is not None:
            state["attention"] = {
                "q_proj": {"weight": self.attention.q_proj.weight, "bias": self.attention.q_proj.bias},
                "k_proj": {"weight": self.attention.k_proj.weight, "bias": self.attention.k_proj.bias},
                "v_proj": {"weight": self.attention.v_proj.weight, "bias": self.attention.v_proj.bias},
                "out_proj": {"weight": self.attention.out_proj.weight, "bias": self.attention.out_proj.bias},
            }
        else:
            state.update(
                {
                    "q_proj": self.q_proj.state_dict(),
                    "k_proj": self.k_proj.state_dict(),
                    "v_proj": self.v_proj.state_dict(),
                    "out_proj": self.out_proj.state_dict(),
                }
            )
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if self.attention is not None and "attention" in state:
            att = state["attention"]
            self.attention.q_proj.weight = att["q_proj"]["weight"]
            self.attention.q_proj.bias = att["q_proj"]["bias"]
            self.attention.k_proj.weight = att["k_proj"]["weight"]
            self.attention.k_proj.bias = att["k_proj"]["bias"]
            self.attention.v_proj.weight = att["v_proj"]["weight"]
            self.attention.v_proj.bias = att["v_proj"]["bias"]
            self.attention.out_proj.weight = att["out_proj"]["weight"]
            self.attention.out_proj.bias = att["out_proj"]["bias"]
        else:
            self.q_proj.load_state_dict(state["q_proj"])
            self.k_proj.load_state_dict(state["k_proj"])
            self.v_proj.load_state_dict(state["v_proj"])
            self.out_proj.load_state_dict(state["out_proj"])
