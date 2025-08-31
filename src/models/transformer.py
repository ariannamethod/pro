"""Tiny transformer-like module with optional adapters and memristor memory."""

from __future__ import annotations

import numpy as np

from adapters.fractal_adapter import FractalAdapter
try:  # pragma: no cover - fallback for different package layouts
    from memory.memristor_cell import MemristorCell
except ModuleNotFoundError:  # pragma: no cover
    from src.memory.memristor_cell import MemristorCell
from transformers.blocks import LightweightMoEBlock


class Transformer:
    """A minimal transformer block supporting fractal adapters."""

    def __init__(
        self,
        dim: int,
        use_fractal_adapter: bool = False,
        fractal_depth: int = 1,
        use_memristor: bool = False,
        use_light_moe: bool = False,
    ) -> None:
        self.dim = dim
        self.use_fractal_adapter = use_fractal_adapter
        self.fractal_depth = fractal_depth
        self.adapter = None
        if use_fractal_adapter:
            self.adapter = FractalAdapter(fractal_depth)
        self.memristor = MemristorCell() if use_memristor else None
        self.light_moe = LightweightMoEBlock(dim) if use_light_moe else None

    def forward(
        self,
        hidden_states: np.ndarray,
        adapters: list[dict[str, float]] | None = None,
    ) -> np.ndarray:
        """Return hidden states enriched with optional features."""

        output = hidden_states
        if self.adapter is not None:
            adapter_vec = self.adapter(hidden_states.shape[-1])
            output = output + adapter_vec
        if self.light_moe is not None:
            output = self.light_moe(output, adapters=adapters)
        if self.memristor is not None:
            resistance = self.memristor.step(float(np.mean(output)))
            output = output * resistance
        return output

    __call__ = forward
