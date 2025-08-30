import json
import os
from typing import Dict, List


class LayerMutator:
    """Track and persist layer modifications."""

    def __init__(self) -> None:
        self.mutations: Dict[str, float] = {}

    def mutate(self, layer: str, factor: float) -> None:
        """Apply a simple multiplicative mutation to a layer."""
        current = self.mutations.get(layer, 1.0)
        self.mutations[layer] = current * factor

    def save(self, directory: str) -> None:
        """Persist mutation state to ``directory``."""
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "mutations.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.mutations, fh)


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
