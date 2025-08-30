"""Simple training loop with layer evaluation and time folding."""
from autoadapt import LayerMutator, MetricMonitor
import numpy as np
from quantum_memory import QuantumMemory
from transformers.time_fold import TimeFoldTransformer


class Trainer:
    """Train model and periodically evaluate layer effectiveness."""

    def __init__(
        self,
        eval_interval: int = 10,
        checkpoint_dir: str = "checkpoints/autoadapt",
        time_fold_steps: int = 1,
    ) -> None:
        self.eval_interval = eval_interval
        self.monitor = MetricMonitor()
        self.mutator = LayerMutator()
        self.step = 0
        self.checkpoint_dir = checkpoint_dir
        self.time_fold_steps = time_fold_steps

    def train_step(self, layer: str, metric: float) -> None:
        """Simulate a training step and record metric for ``layer``."""
        self.monitor.record(metric)
        if self.time_fold_steps > 1:
            self._gradient_echo(metric)
        self.step += 1
        if self.step % self.eval_interval == 0:
            self._evaluate(layer)

    def _evaluate(self, layer: str) -> None:
        avg = self.monitor.average()
        if avg < 0.5:
            # Boost the layer slightly if performance is lacking
            self.mutator.mutate(layer, 1.1)
            self.mutator.save(self.checkpoint_dir)

    # Time folding ------------------------------------------------------
    def _gradient_echo(self, value: float) -> None:
        """Propagate a dummy gradient through future predictions."""

        tf = TimeFoldTransformer(lambda x: x, self.time_fold_steps, QuantumMemory())
        arr = np.array([value], dtype=np.float32)
        tf.forward(arr)
        tf.echo_backward(np.ones_like(arr))
