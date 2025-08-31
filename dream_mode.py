"""Background dreaming mode generating synthetic dialogues."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

import pro_memory
from trainer import Trainer

try:  # Optional YAML support
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

_DEFAULT_DATA = {
    "prompts": [
        "describe a surreal landscape",
        "share a quirky dream snippet",
        "what would clouds say if they could speak?",
    ],
    "responses": [
        "In dreams, the impossible folds into reality.",
        "I wandered through a maze of luminous equations.",
        "Clouds might whisper secrets about changing shapes.",
    ],
}

_DEFAULT_DATA_FILE = Path(__file__).with_name("dream_data.json")

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from pro_engine import ProEngine


def _load_dialogue_data(
    path: str | None = None,
) -> tuple[list[str], list[str]]:
    """Load prompts and responses from JSON or YAML."""
    data_path = Path(path) if path else _DEFAULT_DATA_FILE
    if data_path.is_file():
        with data_path.open("r", encoding="utf-8") as f:
            if data_path.suffix.lower() in {".yml", ".yaml"} and yaml:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        prompts = data.get("prompts", _DEFAULT_DATA["prompts"])
        responses = data.get("responses", _DEFAULT_DATA["responses"])
        return prompts, responses
    return _DEFAULT_DATA["prompts"], _DEFAULT_DATA["responses"]


def _simulate_dialogue(
    turns: int = 3, data_path: str | None = None
) -> list[str]:
    """Generate a simple alternating dialogue."""
    prompts, responses = _load_dialogue_data(data_path)
    dialogue: list[str] = []
    for _ in range(turns):
        dialogue.append(random.choice(prompts))
        dialogue.append(random.choice(responses))
    return dialogue


async def run(
    engine: ProEngine, turns: int = 3, data_path: str | None = None
) -> None:
    """Generate dialogue and route through the training loop."""
    dialogue = _simulate_dialogue(turns, data_path=data_path)
    trainer = Trainer()
    trainer.evolve("dream", dialogue, metric=1.0)
    for line in dialogue:
        await pro_memory.add_message(line, tag="dream")
