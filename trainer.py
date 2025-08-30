"""Simple training loop with layer evaluation."""
from autoadapt import LayerMutator, MetricMonitor


class Trainer:
    """Train model and periodically evaluate layer effectiveness."""

    def __init__(
        self,
        eval_interval: int = 10,
        checkpoint_dir: str = "checkpoints/autoadapt",
    ) -> None:
        self.eval_interval = eval_interval
        self.monitor = MetricMonitor()
        self.mutator = LayerMutator()
        self.step = 0
        self.checkpoint_dir = checkpoint_dir

    def train_step(self, layer: str, metric: float) -> None:
        """Simulate a training step and record metric for ``layer``."""
        self.monitor.record(metric)
        self.step += 1
        if self.step % self.eval_interval == 0:
            self._evaluate(layer)

    def _evaluate(self, layer: str) -> None:
        avg = self.monitor.average()
        if avg < 0.5:
            # Boost the layer slightly if performance is lacking
            self.mutator.mutate(layer, 1.1)
            self.mutator.save(self.checkpoint_dir)
