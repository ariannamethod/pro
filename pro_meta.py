import json
import os
import asyncio
from compat import to_thread
import random
from typing import Any, Dict, List, Optional

META_PATH = "pro_meta.json"

_history: List[Dict[str, Any]] = []
_best_params: Dict[str, float] = {
    "chaos_factor": 0.0,
    "similarity_threshold": 0.3,
}
_recompute_task: Optional[asyncio.Task] = None


def _load() -> None:
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            _history.extend(data.get("history", []))
            _best_params.update(data.get("best_params", {}))
        except Exception:
            pass


def _save() -> None:
    with open(META_PATH, "w", encoding="utf-8") as fh:
        json.dump({"history": _history, "best_params": _best_params}, fh)


async def _recompute() -> None:
    if not _history:
        return
    best = min(
        _history, key=lambda h: h["metrics"].get("perplexity", float("inf"))
    )
    evolved: Dict[str, float] = {}
    for k, v in best["params"].items():
        noise = random.uniform(-0.05, 0.05)
        evolved[k] = max(0.0, min(1.0, v + noise))
    _best_params.update(evolved)
    await to_thread(_save)


async def wait_recompute() -> None:
    global _recompute_task
    if _recompute_task is not None:
        try:
            await _recompute_task
        except asyncio.CancelledError:
            pass
        _recompute_task = None


def update(metrics: Dict[str, float], params: Dict[str, float]) -> None:
    global _recompute_task
    _history.append({"metrics": metrics, "params": params})
    _save()
    _recompute_task = asyncio.create_task(_recompute())


def best_params() -> Dict[str, float]:
    return dict(_best_params)


_load()
