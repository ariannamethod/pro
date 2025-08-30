from .blocks import DynamicContextGate
from .modeling_transformer import MemoryAttention
from .quantum_attention import QuantumAttention
from .time_fold import TimeFoldTransformer

__all__ = [
    "DynamicContextGate",
    "MemoryAttention",
    "QuantumAttention",
    "TimeFoldTransformer",
    "get_attention",
]


def get_attention(kind: str):
    """Return an attention class by *kind*.

    Parameters
    ----------
    kind : str
        ``"quantum"`` for :class:`QuantumAttention`, ``"memory"`` for
        :class:`MemoryAttention`.
    """

    if kind == "quantum":
        return QuantumAttention
    if kind == "memory":
        return MemoryAttention
    raise ValueError(f"Unknown attention kind: {kind}")
