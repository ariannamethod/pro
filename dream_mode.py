"""Background dreaming mode generating synthetic dialogues."""

import random
from pathlib import Path
from typing import List, Optional, Sequence

import pro_memory
from trainer import Trainer

DEFAULT_DATASET = "datasets/smalltalk.txt"


def _load_corpus(path: str) -> List[str]:
    """Load a list of utterances from ``path``.

    Each non-empty line in the file becomes a candidate utterance. If the file
    does not exist, an empty list is returned so callers can handle the lack of
    data gracefully.
    """

    p = Path(path)
    if not p.exists():
        return []
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


def _simulate_dialogue(
    turns: int = 3, corpus: Optional[Sequence[str]] = None
) -> List[str]:
    """Generate a simple alternating dialogue from a dynamic corpus."""

    lines: Sequence[str]
    if corpus is None:
        lines = _load_corpus(DEFAULT_DATASET)
    else:
        lines = corpus
    if not lines:
        return []
    dialogue: List[str] = []
    for _ in range(turns):
        dialogue.append(random.choice(lines))
        dialogue.append(random.choice(lines))
    return dialogue


async def run(
    engine: "ProEngine", turns: int = 3, dataset_path: Optional[str] = None
) -> None:
    """Generate dialogue and route through the training loop."""

    corpus = _load_corpus(dataset_path or DEFAULT_DATASET)
    dialogue = _simulate_dialogue(turns, corpus)
    trainer = Trainer()
    trainer.evolve("dream", dialogue, metric=1.0)
    for line in dialogue:
        await pro_memory.add_message(line, tag="dream")
