"""Utilities for saving LoRA adapters for hot swapping."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from autoadapt import LoRALayer
from lora_utils import prune_and_merge


def save_hotswap_lora(layers: Dict[str, LoRALayer], path: str | Path) -> None:
    """Persist LoRA layers in ``HotSwapLoRAAdapter`` compatible format.

    Parameters
    ----------
    layers:
        Mapping of layer name to :class:`autoadapt.LoRALayer` instances.
    path:
        Destination file. Parent directories are created automatically.
    """

    merged = prune_and_merge(layers)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(merged, fh)
