from .blocks import DynamicContextGate, ResonantDropout
from .modeling_transformer import MemoryAttention, register_kernel
from .quantum_attention import QuantumAttention
from .time_fold import TimeFoldTransformer
from .quant4_infer import Quant4Linear, load_quant4

__all__ = [
    "DynamicContextGate",
    "ResonantDropout",
    "MemoryAttention",
    "register_kernel",
    "QuantumAttention",
    "TimeFoldTransformer",
    "Quant4Linear",
    "load_quant4",
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
