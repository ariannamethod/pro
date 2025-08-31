from __future__ import annotations

from typing import Dict, List, Optional


class WeaknessDetector:
    """Detects weaknesses in conversation logs."""

    def detect(self, conversations: List[str]) -> List[str]:
        """Return a list of weakness descriptions found in ``conversations``.

        The implementation is deliberately lightweight; in a production
        setting this could use NLP heuristics or model-based analysis.
        """
        weaknesses: List[str] = []
        for conv in conversations:
            if "??" in conv:
                weaknesses.append("unclear response")
            if len(conv.split()) < 3:
                weaknesses.append("short reply")
        return weaknesses


class MetaOptimizer:
    """Suggest parameter updates based on detected weaknesses."""

    def optimize(
        self, weaknesses: List[str], params: Dict[str, float]
    ) -> Dict[str, float]:
        """Return a mapping of parameter names to update deltas.

        The strategy is intentionally simple: if any weaknesses are
        detected, each parameter receives an additive boost proportional to
        the number of weaknesses. A production implementation could employ
        gradient-based optimisation or reinforcement learning.
        """
        deltas: Dict[str, float] = {}
        if not weaknesses:
            return deltas
        strength = 0.1 * len(weaknesses)
        for name in params:
            deltas[name] = strength
        return deltas


class SelfFineTuner:
    """Orchestrate self-reflection and meta-optimisation cycle."""

    def __init__(self, model: Optional[object] = None) -> None:
        self.model = model

    def run(self, conversations: List[str], params: Dict[str, float]) -> Dict[str, float]:
        """Detect weaknesses and propose parameter updates.

        Returns a dictionary of parameter deltas suggested by the
        :class:`MetaOptimizer`.
        """
        detector = WeaknessDetector()
        weaknesses = detector.detect(conversations)
        optimizer = MetaOptimizer()
        deltas = optimizer.optimize(weaknesses, params)

        # After generating parameter updates, blend the weights of linked
        # layers so their gradients resonate with each other.  This uses the
        # ``fractal_links`` configuration defined in :mod:`trainer`.
        if self.model is not None:
            try:
                from trainer import fractal_links  # local import to avoid cycle
            except Exception:  # pragma: no cover - in case trainer is missing
                fractal_links = []  # type: ignore[assignment]
            resonance = getattr(self.model, "resonance", 0.5)
            for left, right in fractal_links:
                w_left = getattr(self.model, left, None)
                w_right = getattr(self.model, right, None)
                if w_left is None or w_right is None:
                    continue
                new_left = (1 - resonance) * w_left + resonance * w_right
                new_right = (1 - resonance) * w_right + resonance * w_left
                setattr(self.model, left, new_left)
                setattr(self.model, right, new_right)

        return deltas
