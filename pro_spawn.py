import copy
from typing import Dict

from pro_tune import train


def clone_weights(state: Dict) -> Dict:
    """Deep-copy ``state`` to create an isolated clone of model weights."""
    return copy.deepcopy(state)


def fine_tune(state: Dict, dataset_path: str) -> Dict:
    """Fine-tune *state* on data located at *dataset_path*.

    Parameters
    ----------
    state:
        Model state to update.
    dataset_path:
        Path to training dataset.
    """
    return train(state, dataset_path)


def create_specialist(state: Dict, dataset_path: str) -> Dict:
    """Clone ``state`` and fine-tune the clone on ``dataset_path``.

    This utility is used to spawn a specialist model for a specific task or
    topic.  The returned state is the tuned specialist.
    """
    specialist = clone_weights(state)
    return fine_tune(specialist, dataset_path)
