import numpy as np
import numpy.typing as npt

from morphology import encode as encode_morph
from memory.memory_graph import GraphRetriever


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

    def __call__(
        self, x: npt.NDArray[np.complex64]
    ) -> npt.NDArray[np.complex64]:
        if x.ndim != 2:
            raise ValueError("ResonantDropout expects a 2D array")
        dim = x.shape[1]
        idx = np.arange(dim)
        probs = np.sin(self.pos_freq * idx)
        probs = np.clip(probs, 0.0, 1.0)
        mask = (self.rng.random(dim) >= probs).astype(x.dtype)
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
        self.bias = np.zeros(dim, dtype=np.complex64)
        self.dropout = ResonantDropout(pos_freq, seed)

    def __call__(
        self, context: npt.NDArray[np.complex64]
    ) -> npt.NDArray[np.complex64]:
        """Apply the gating mechanism to *context*."""
        context = self.dropout(context)
        mean_ctx = context.mean(axis=0)
        gate = 1.0 / (1.0 + np.exp(-(mean_ctx.real + self.bias.real)))
        return context * gate.astype(context.dtype)

    # Saving/loading helpers -------------------------------------------------
    def state_dict(self) -> dict:
        return {"bias": self.bias}

    def load_state_dict(self, state: dict) -> None:
        if "bias" in state:
            self.bias = state["bias"]


class HoloMemoryGate:
    """Generate a holographic vector from recent dialogue messages."""

    def __init__(
        self, retriever: GraphRetriever, dim: int, n: int = 3
    ) -> None:
        self.retriever = retriever
        self.dim = dim
        self.n = n

    def __call__(self, dialogue_id: str) -> npt.NDArray[np.complex64]:
        if not hasattr(self.retriever, "recent_messages"):
            return np.zeros(self.dim, dtype=np.complex64)
        msgs = self.retriever.recent_messages(dialogue_id, self.n)
        if not msgs:
            return np.zeros(self.dim, dtype=np.complex64)
        vec = np.zeros(self.dim, dtype=np.float32)
        for msg in msgs:
            vec += encode_morph(msg, self.dim)
        vec /= len(msgs)
        phase = np.linspace(0, np.pi, self.dim, dtype=np.float32)
        holo = vec * np.exp(1j * phase)
        return holo.astype(np.complex64)


def amplitude_attention(
    query: npt.NDArray[np.complex64],
    key: npt.NDArray[np.complex64],
) -> npt.NDArray[np.complex64]:
    """Return amplitude-based attention weights."""
    scores = (query @ key.conj().T) / np.sqrt(key.shape[-1])
    return np.abs(scores).astype(np.complex64)


def phase_attention(
    query: npt.NDArray[np.complex64],
    key: npt.NDArray[np.complex64],
) -> npt.NDArray[np.complex64]:
    """Return phase-based attention weights."""
    scores = (query @ key.conj().T) / np.sqrt(key.shape[-1])
    return np.exp(1j * np.angle(scores)).astype(np.complex64)


def wave_attention(
    query: npt.NDArray[np.complex64],
    key: npt.NDArray[np.complex64],
    value: npt.NDArray[np.complex64],
) -> npt.NDArray[np.complex64]:
    """Combine amplitude and phase attention into a complex output."""
    amp = amplitude_attention(query, key)
    phase = phase_attention(query, key)
    weights = amp * phase
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ value
