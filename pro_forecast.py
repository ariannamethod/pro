import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Any

import numpy as np

import pro_predict


@dataclass
class ForecastNode:
    text: str
    prob: float
    novelty: float
    children: List["ForecastNode"] = field(default_factory=list)


def _softmax(logits: Dict[str, float]) -> Dict[str, float]:
    values = np.array(list(logits.values()))
    exp = np.exp(values - values.max())
    probs = exp / exp.sum()
    return {w: float(p) for w, p in zip(logits.keys(), probs)}


def simulate_paths(seeds: List[str], depth: int = 2) -> ForecastNode:
    """Simulate possible continuations using a tiny self-attention model.

    Parameters
    ----------
    seeds:
        Starting tokens for the simulation.
    depth:
        How many steps to expand the tree.

    Returns
    -------
    ForecastNode
        Root node representing *seeds* with nested children branches.
    """

    pro_predict._ensure_vectors()
    vocab = list(pro_predict._VECTORS.keys())

    def _expand(tokens: List[str], remaining: int, prob: float) -> ForecastNode:
        node = ForecastNode(" ".join(tokens), prob, 1.0 - prob)
        if remaining == 0:
            return node
        logits = pro_predict.transformer_logits(tokens, vocab)
        probs = _softmax(logits)
        ordered = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        for word, p in ordered:
            child = _expand(tokens + [word], remaining - 1, prob * p)
            child.novelty = 1.0 - p
            node.children.append(child)
        return node

    return _expand(seeds, depth, 1.0)
