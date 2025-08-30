from typing import Sequence
from autoadapt import LayerMutator, MetricMonitor


class MiniModel:
    """Generate and evaluate a lightweight model variant."""

    def __init__(self, layer: str, use_lora: bool = False) -> None:
        self.layer = layer
        self.mutator = LayerMutator(use_lora=use_lora)
        self.monitor = MetricMonitor()
        # Start with a small mutation to explore the space
        self.mutator.mutate(layer, 0.9)

    def train(self, dialogue_part: Sequence[str]) -> None:
        """Record metrics for a slice of dialogue."""
        for turn in dialogue_part:
            metric = len(turn) / (len(turn) + 1)
            self.monitor.record(metric)

    def score(self) -> float:
        """Return the average metric for the mini-model."""
        return self.monitor.average()

    def distill(self, target: LayerMutator) -> None:
        """Copy learned mutations into ``target``."""
        target.mutations.update(self.mutator.mutations)
