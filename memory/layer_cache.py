"""Caching helpers for assembled micro layers."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict


_CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def cache_layers(macros: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Return assembled micro layers for ``macros`` with caching.

    The macro specification is hashed to derive a cache file name.  If the
    corresponding file exists the layers are loaded from disk, otherwise the
    layers are constructed via :func:`pro_spawn.construct_layers` and written
    back to the cache directory.
    """

    os.makedirs(_CACHE_DIR, exist_ok=True)
    key = hashlib.sha1(json.dumps(macros, sort_keys=True).encode("utf-8")).hexdigest()
    path = os.path.join(_CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    from pro_spawn import construct_layers  # Lazy import to avoid heavy deps

    layers = construct_layers(macros)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(layers, fh)
    return layers
