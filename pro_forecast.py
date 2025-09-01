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

    asyncio.run(pro_predict._ensure_vectors())
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


def backpropagate_forecast(node: ForecastNode) -> None:
    """Propagate novelty backwards to update prediction weights.

    The update scales the learning rate by ``node.novelty`` so that
    highly unexpected branches have a proportionally larger influence on
    the underlying :class:`pro_predict.MiniSelfAttention` model. The
    function walks the entire subtree rooted at ``node`` and applies a
    small training step for each node that represents at least one
    prediction (i.e. ``len(tokens) > 1``).
    """

    asyncio.run(pro_predict._ensure_vectors())
    vocab = list(pro_predict._VECTORS.keys())
    model_key = tuple(vocab)
    if model_key not in pro_predict._TRANSFORMERS:
        pro_predict._TRANSFORMERS[model_key] = pro_predict.MiniSelfAttention(vocab)
    model = pro_predict._TRANSFORMERS[model_key]

    tokens = node.text.split()
    if len(tokens) > 1 and node.novelty > 0:
        context, target = tokens[:-1], tokens[-1]
        lr = 0.1 * node.novelty
        model.train_step([context], [target], lr=lr)

    for child in node.children:
        backpropagate_forecast(child)
