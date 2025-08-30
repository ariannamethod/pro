import numpy as np


class DynamicContextGate:
    """Simple dynamic gating for attention contexts.

    The gate computes a sigmoid over the mean context vector plus a learnable
    bias. The resulting gate scales the context vectors element-wise.
    """

    def __init__(self, dim: int) -> None:
        self.bias = np.zeros(dim, dtype=np.float32)

    def __call__(self, context: np.ndarray) -> np.ndarray:
        """Apply the gating mechanism to *context*.

        Parameters
        ----------
        context:
            Array of shape ``(seq_len, dim)`` representing the attention
            context vectors.
        """
        mean_ctx = context.mean(axis=0)
        gate = 1.0 / (1.0 + np.exp(-(mean_ctx + self.bias)))
        return context * gate

    # Saving/loading helpers -------------------------------------------------
    def state_dict(self) -> dict:
        return {"bias": self.bias}

    def load_state_dict(self, state: dict) -> None:
        if "bias" in state:
            self.bias = state["bias"]
