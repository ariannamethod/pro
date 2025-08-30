"""Tiny transformer-like module with optional fractal adapter."""

from __future__ import annotations

import numpy as np

from adapters.fractal_adapter import FractalAdapter


class Transformer:
    """A minimal transformer block supporting fractal adapters."""

    def __init__(
        self,
        dim: int,
        use_fractal_adapter: bool = False,
        fractal_depth: int = 1,
    ) -> None:
        self.dim = dim
        self.use_fractal_adapter = use_fractal_adapter
        self.fractal_depth = fractal_depth
        self.adapter = None
        if use_fractal_adapter:
            self.adapter = FractalAdapter(fractal_depth)

    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """Return hidden states enriched with adapter signal when enabled."""

        if self.adapter is None:
            return hidden_states
        adapter_vec = self.adapter(hidden_states.shape[-1])
        return hidden_states + adapter_vec

    __call__ = forward
