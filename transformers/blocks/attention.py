import numpy as np


class ResonantDropout:
    """Dropout with sinusoidal per-dimension probabilities.

    Each feature dimension ``idx`` is dropped with probability
    ``sin(pos_freq * idx)``. Probabilities are clipped to ``[0, 1]`` to make
    them valid dropout rates. A dedicated random number generator can be
    provided for reproducibility.
    """

    def __init__(self, pos_freq: float, seed: int | None = None) -> None:
        self.pos_freq = pos_freq
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError("ResonantDropout expects a 2D array")
        dim = x.shape[1]
        idx = np.arange(dim)
        probs = np.sin(self.pos_freq * idx)
        probs = np.clip(probs, 0.0, 1.0)
        mask = self.rng.random(dim) >= probs
        return x * mask


class DynamicContextGate:
    """Simple dynamic gating for attention contexts.

    The gate computes a sigmoid over the mean context vector plus a learnable
    bias. Before gating, the context is processed by :class:`ResonantDropout`.
    The resulting gate scales the context vectors element-wise.
    """

    def __init__(
        self, dim: int, pos_freq: float = 1.0, seed: int | None = None
    ) -> None:
        self.bias = np.zeros(dim, dtype=np.float32)
        self.dropout = ResonantDropout(pos_freq, seed)

    def __call__(self, context: np.ndarray) -> np.ndarray:
        """Apply the gating mechanism to *context*."""
        context = self.dropout(context)
        mean_ctx = context.mean(axis=0)
        gate = 1.0 / (1.0 + np.exp(-(mean_ctx + self.bias)))
        return context * gate

    # Saving/loading helpers -------------------------------------------------
    def state_dict(self) -> dict:
        return {"bias": self.bias}

    def load_state_dict(self, state: dict) -> None:
        if "bias" in state:
            self.bias = state["bias"]
