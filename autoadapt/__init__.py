import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List
import textwrap


@dataclass
class LoRALayer:
    """Minimal representation of a LoRA adapter layer."""

    name: str
    rank: int
    alpha: float
    matrix_a: List[List[float]]
    matrix_b: List[List[float]]

    def save(self, path: str) -> None:
        """Serialize layer parameters to *path* as JSON."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh)

    @classmethod
    def load(cls, path: str) -> "LoRALayer":
        """Load layer parameters from *path*."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(**data)


class LayerMutator:
    """Track and persist layer modifications and optional LoRA adapters."""

    def __init__(self, use_lora: bool = False) -> None:
        self.mutations: Dict[str, float] = {}
        self.use_lora = use_lora
        self.lora_layers: Dict[str, LoRALayer] = {}

    def mutate(self, layer: str, factor: float) -> None:
        """Apply a simple multiplicative mutation to a layer."""
        current = self.mutations.get(layer, 1.0)
        self.mutations[layer] = current * factor

    # -- LoRA -----------------------------------------------------------
    def add_lora_layer(self, layer: LoRALayer) -> None:
        """Register a :class:`LoRALayer` for persistence."""
        if self.use_lora:
            self.lora_layers[layer.name] = layer

    def save_lora(self, directory: str) -> None:
        """Save all registered LoRA layers to *directory*."""
        if not self.use_lora or not self.lora_layers:
            return
        os.makedirs(directory, exist_ok=True)
        for layer in self.lora_layers.values():
            path = os.path.join(directory, f"{layer.name}.json")
            layer.save(path)

    def load_lora(self, directory: str) -> None:
        """Load LoRA layers from *directory* if present."""
        if not self.use_lora or not os.path.isdir(directory):
            return
        for fname in os.listdir(directory):
            if fname.endswith(".json"):
                layer = LoRALayer.load(os.path.join(directory, fname))
                self.lora_layers[layer.name] = layer

    # -- Persistence ----------------------------------------------------
    def save(self, directory: str) -> None:
        """Persist mutation state to ``directory``."""
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "mutations.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.mutations, fh)
        if self.use_lora:
            self.save_lora(os.path.join(directory, "lora"))

    def load(self, directory: str) -> None:
        """Load mutation state and LoRA layers from ``directory``."""
        path = os.path.join(directory, "mutations.json")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                self.mutations = json.load(fh)
        if self.use_lora:
            self.load_lora(os.path.join(directory, "lora"))


class MetricMonitor:
    """Keep a sliding window of metrics and compute averages."""

    def __init__(self, window: int = 10) -> None:
        self.window = window
        self.values: List[float] = []

    def record(self, value: float) -> None:
        self.values.append(value)
        if len(self.values) > self.window:
            self.values.pop(0)

    def average(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0


def generate_block_code(name: str, scale: float = 1.0) -> str:
    """Return source code for a tiny processing block.

    The generated code defines a class with ``forward`` method that multiplies
    its input by ``scale``.  The resulting string can be compiled and loaded as
    a module using :meth:`pro_engine.ProEngine.load_generated_block`.

    Parameters
    ----------
    name:
        Name of the class to generate.
    scale:
        Factor applied to the input in ``forward``.
    """

    template = f"""class {name}:
    def __init__(self, scale={scale}):
        self.scale = scale

    def forward(self, x):
        return x * self.scale
"""

    return textwrap.dedent(template)


__all__ = [
    "LoRALayer",
    "LayerMutator",
    "MetricMonitor",
    "generate_block_code",
]
