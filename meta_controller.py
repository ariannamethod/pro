import json
import random
from typing import Dict, List, Optional

STATE_PATH = "pro_state.json"


class MetaController:
    """Epsilon-greedy controller tuning layer choices via :mod:`pro_metrics`.

    The controller treats each candidate architecture as an arm in a multi-armed
    bandit.  After each response the negative perplexity of the model is used as
    reward.  Value estimates are updated online so that better architectures are
    chosen more frequently over time.
    """

    def __init__(
        self,
        architectures: Optional[List[Dict[str, int]]] = None,
        epsilon: float = 0.1,
    ) -> None:
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

    def update(self, metrics: Dict[str, float]) -> None:
        """Update value estimates based on observed metrics."""

        if self._last_index is None:
            return
        reward = -float(metrics.get("perplexity", 0.0))
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
