"""Simple training loop with layer evaluation and time folding."""
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from autoadapt import LayerMutator, MetricMonitor
from pro_evo import MiniModel
from quantum_memory import QuantumMemory
from transformers.time_fold import TimeFoldTransformer
from pro_forecast import simulate_paths, backpropagate_forecast
from resonance.p2p_resonance import P2PResonance
from self_reflect.trainer import SelfFineTuner

# Pairs of layer names whose gradients should mirror each other during
# self-reflection.  Layers listed here will have their weights blended
# symmetrically by :class:`~self_reflect.SelfFineTuner` after each cycle.
fractal_links: List[Tuple[str, str]] = []


class Trainer:
    """Train model and periodically evaluate layer effectiveness."""

    def __init__(
        self,
        eval_interval: int = 10,
        checkpoint_dir: str = "checkpoints/autoadapt",
        time_fold_steps: int = 1,
        use_lora: bool = False,
        resonance_peer: P2PResonance | None = None,
        sync_target: tuple[str, int] | None = None,
        sync_interval: int = 10,
    ) -> None:
        self.eval_interval = eval_interval
        self.monitor = MetricMonitor()
        self.mutator = LayerMutator(use_lora=use_lora)
        if use_lora:
            self.mutator.load(checkpoint_dir)
        self.step = 0
        self.checkpoint_dir = checkpoint_dir
        self.time_fold_steps = time_fold_steps
        self.resonance_peer = resonance_peer
        self.sync_target = sync_target
        self.sync_interval = sync_interval
        self.params = resonance_peer.params if resonance_peer else {}

    def train_step(
        self,
        layer: str,
        metric: float,
        conversations: Optional[List[str]] = None,
    ) -> None:
        """Simulate a training step and record metric for ``layer``."""
        forecast = simulate_paths([layer])
        backpropagate_forecast(forecast)
        self.monitor.record(metric)
        if self.resonance_peer:
            self.resonance_peer.queue_update({layer: metric})
            if (
                self.sync_target
                and self.sync_interval > 0
                and self.step % self.sync_interval == 0
            ):
                self.resonance_peer.exchange(*self.sync_target)
        if self.time_fold_steps > 1:
            self._gradient_echo(metric)
        self.step += 1
        if self.step % self.eval_interval == 0:
            self._evaluate(layer)
            self._meta_optimize(conversations or [])

    def evolve(self, layer: str, dialogue: list[str], metric: float) -> None:
        """Train a mini-model on a slice of dialogue and distill weights."""
        mini = MiniModel(layer, use_lora=self.mutator.use_lora)
        half = max(1, len(dialogue) // 2)
        mini.train(dialogue[:half])
        if mini.score() < metric:
            mini.distill(self.mutator)

    def _evaluate(self, layer: str) -> None:
        avg = self.monitor.average()
        if avg < 0.5:
            # Boost the layer slightly if performance is lacking
            self.mutator.mutate(layer, 1.1)
            self.mutator.save(self.checkpoint_dir)

    def _meta_optimize(self, conversations: List[str]) -> None:
        """Invoke :class:`SelfFineTuner` and apply suggested updates."""
        tuner = SelfFineTuner()
        meta_dir = Path("logs/meta")
        meta_dir.mkdir(parents=True, exist_ok=True)
        before = self.params.copy()
        before_file = meta_dir / f"epoch_{self.step}_before.json"
        after_file = meta_dir / f"epoch_{self.step}_after.json"
        with before_file.open("w", encoding="utf-8") as fh:
            json.dump(before, fh)
        deltas = tuner.run(conversations, self.params)
        for name, delta in deltas.items():
            self.params[name] = self.params.get(name, 0.0) + delta
        with after_file.open("w", encoding="utf-8") as fh:
            json.dump(self.params, fh)

    # Time folding ------------------------------------------------------
    def _gradient_echo(self, value: float) -> None:
        """Propagate a dummy gradient through future predictions."""

        tf = TimeFoldTransformer(
            lambda x: x, self.time_fold_steps, QuantumMemory()
        )
        arr = np.array([value], dtype=np.float32)
        tf.forward(arr)
        tf.echo_backward(np.ones_like(arr))
