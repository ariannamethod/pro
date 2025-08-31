import numpy as np
import numpy.typing as npt
from typing import Dict, List, Optional


def _softmax(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Numerically stable softmax."""

    e = np.exp(x - np.max(x))
    return e / e.sum()


class LightweightMoEBlock:
    """Minimal mixture-of-experts block with optional adapter scaling.

    The block routes inputs to up to ``top_k`` of ``num_experts`` experts using
    a lightweight gating network inspired by DeepSeek's mixture-of-experts
    design.  Adapter weights from :class:`ProEngine` can optionally scale the
    output, enabling external specialization without retraining the experts.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 4,
        top_k: int = 1,
        seed: int | None = None,
    ) -> None:
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = max(1, min(top_k, num_experts))
        rng = np.random.default_rng(seed)
        self.experts = rng.standard_normal((num_experts, dim, dim), dtype=np.float32)
        self.gate = rng.standard_normal((dim, num_experts), dtype=np.float32)

    def __call__(
        self,
        x: npt.NDArray[np.float32],
        adapters: Optional[List[Dict[str, float]]] = None,
    ) -> npt.NDArray[np.float32]:
        """Apply gated experts to *x* and scale with *adapters*.

        Parameters
        ----------
        x:
            Input vector of shape ``(dim,)``.
        adapters:
            Optional list of adapter weight dictionaries provided by
            :class:`ProEngine`.  The mean of all bias values is used as a
            multiplicative scale on the expert output.
        """

        if x.shape != (self.dim,):
            raise ValueError(f"expected input of shape ({self.dim},)")

        logits = x @ self.gate
        probs = _softmax(logits)

        if self.top_k == 1:
            expert_idx = int(np.argmax(probs))
            output = self.experts[expert_idx] @ x
            output = output * probs[expert_idx]
        else:
            top_idx = np.argpartition(probs, -self.top_k)[-self.top_k:]
            weights = probs[top_idx]
            weights = weights / weights.sum()
            output = np.zeros(self.dim, dtype=np.float32)
            for w, idx in zip(weights, top_idx):
                output += w * (self.experts[int(idx)] @ x)

        if adapters:
            bias_sum = sum(sum(a.values()) for a in adapters)
            scale = 1.0 + bias_sum / (len(adapters) or 1)
            output = output * scale
        return output
