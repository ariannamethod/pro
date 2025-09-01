"""Utilities for working with LoRA adapters."""

from __future__ import annotations

from typing import Dict

import numpy as np

from autoadapt import LoRALayer


def prune_and_merge(layers: Dict[str, LoRALayer], threshold: float = 0.01) -> Dict[str, list[list[float]]]:
    """Prune tiny values and merge LoRA layers into weight deltas.

    Parameters
    ----------
    layers:
        Mapping of layer name to :class:`autoadapt.LoRALayer` objects.
    threshold:
        Values with absolute magnitude below this threshold are removed.

    Returns
    -------
    Dict[str, list[list[float]]]
        Mapping of layer name to pruned weight delta matrices represented as
        nested lists.
    """
    merged: Dict[str, list[list[float]]] = {}
    for name, layer in layers.items():
        a = np.array(layer.matrix_a)
        b = np.array(layer.matrix_b)
        delta = (a @ b) * (layer.alpha / layer.rank)
        delta[np.abs(delta) < threshold] = 0.0
        merged[name] = delta.tolist()
    return merged
