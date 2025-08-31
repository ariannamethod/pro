"""Simple memristor cell module with continuous resistance."""
from __future__ import annotations


class MemristorCell:
    """Simulated memristor cell maintaining a continuous resistance state."""

    def __init__(self, resistance: float = 1.0, alpha: float = 0.1) -> None:
        self.resistance = resistance
        self.alpha = alpha

    def step(self, input_signal: float) -> float:
        """Update resistance toward the input signal and return new value."""
        self.resistance += self.alpha * (input_signal - self.resistance)
        return self.resistance

    __call__ = step
