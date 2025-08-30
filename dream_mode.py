"""Background dreaming mode generating synthetic dialogues."""

import random
from typing import List

import pro_memory
from trainer import Trainer

_PROMPTS = [
    "hello there",
    "how are you",
    "tell me a story",
    "what is your favorite color",
    "do you like music",
]

_RESPONSES = [
    "I am just code, but I am functioning well",
    "once upon a time a small bot dreamed",
    "I like all colors equally",
    "music is delightful even for code",
    "thanks for asking",
]


def _simulate_dialogue(turns: int = 3) -> List[str]:
    """Generate a simple alternating dialogue."""
    dialogue: List[str] = []
    for _ in range(turns):
        dialogue.append(random.choice(_PROMPTS))
        dialogue.append(random.choice(_RESPONSES))
    return dialogue


async def run(engine: "ProEngine", turns: int = 3) -> None:
    """Generate dialogue and route through the training loop."""
    dialogue = _simulate_dialogue(turns)
    trainer = Trainer()
    trainer.evolve("dream", dialogue, metric=1.0)
    for line in dialogue:
        await pro_memory.add_message(line, tag="dream")
