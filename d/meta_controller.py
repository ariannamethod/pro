import json
import random
import math
import os
from collections import deque
from typing import Dict, List, Optional

import autoadapt

STATE_PATH = "pro_state.json"
SPAWN_CONFIG = os.path.join(os.path.dirname(__file__), "autoadapt", "meta_spawn.json")


class MetaController:
    """Unified controller managing architecture search and self-tuning."""

    def __init__(
        self,
        engine,
        architectures: Optional[List[Dict[str, int]]] = None,
        epsilon: float = 0.1,
        window: int = 5,
        tolerance: float = 0.1,
    ) -> None:
        self.engine = engine
        if architectures is None:
            architectures = [
                {"layers": 1},
                {"layers": 2},
                {"layers": 3},
            ]
        self.architectures = architectures
        self.epsilon = epsilon
        self.values = [0.0 for _ in architectures]
        self.counts = [0 for _ in architectures]
        self._last_index: Optional[int] = None
        self.window = window
        self.tolerance = tolerance
        self._history: deque[float] = deque(maxlen=window)
        self._baseline: Optional[float] = None
        self._load_autoadapt_blocks()

    # -- Autoadapt ----------------------------------------------------
    def _load_autoadapt_blocks(self) -> None:
        """Load and attach autoadapt blocks defined in ``meta_spawn.json``."""

        if not os.path.isfile(SPAWN_CONFIG):
            return
        try:
            with open(SPAWN_CONFIG, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
        except Exception:
            return
        for spec in cfg.get("blocks", []):
            name = spec.get("name")
            scale = spec.get("scale", 1.0)
            if not name:
                continue
            try:
                code = autoadapt.generate_block_code(name, scale)
                self.engine.load_generated_block(code, name)
            except Exception:
                continue

    def select(self) -> Dict[str, int]:
        """Select an architecture using an epsilon-greedy policy."""

        if not self.architectures:
            return {}
        if random.random() < self.epsilon:
            idx = random.randrange(len(self.architectures))
        else:
            max_val = max(self.values)
            best = [i for i, v in enumerate(self.values) if v == max_val]
            idx = random.choice(best)
        self._last_index = idx
        return self.architectures[idx]

    async def update(self, metrics: Dict[str, float]) -> None:
        """Update metrics, trigger tuning, and refine architecture values."""

        perplexity = metrics.get("perplexity")
        if perplexity is None or math.isnan(perplexity):
            return

        self._history.append(float(perplexity))
        if len(self._history) >= self.window:
            avg = sum(self._history) / len(self._history)
            if self._baseline is None:
                self._baseline = avg
            elif avg > self._baseline * (1 + self.tolerance):
                try:
                    self.engine._start_tune_worker()
                    self.engine.dataset_queue.put_nowait(None)
                except Exception:
                    pass
                self._baseline = avg

        if self._last_index is None:
            return
        reward = -float(perplexity)
        idx = self._last_index
        self.counts[idx] += 1
        lr = 1.0 / self.counts[idx]
        self.values[idx] += lr * (reward - self.values[idx])

    def persist(self, architecture: Dict[str, int], path: str = STATE_PATH) -> None:
        """Append the chosen architecture to ``pro_state.json``."""

        try:
            with open(path, "r", encoding="utf-8") as fh:
                state = json.load(fh)
        except Exception:
            state = {}
        archs = state.setdefault("architectures", [])
        archs.append(architecture)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(state, fh)
