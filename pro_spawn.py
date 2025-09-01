import copy
from typing import Dict, Any

from pro_tune import train


def construct_layers(macros: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, list[float]]]:
    """Build simple micro-layer parameter dictionaries from macros.

    Parameters
    ----------
    macros:
        Mapping of layer name to a specification.  Each specification may
        contain ``in`` and ``out`` sizes describing the expected dimensions of
        the layer.  Additional keys are ignored.

    Returns
    -------
    Dict[str, Dict[str, list[float]]]
        Mapping of layer name to dictionaries with ``weights`` and ``bias``
        lists.  The weights are initialised with zeros and shaped according to
        the provided dimensions.

    Notes
    -----
    This lightweight constructor serves as a placeholder for a more elaborate
    assembly process.  It enables downstream components to request concrete
    layer parameters from compact macro descriptions.
    """

    layers: Dict[str, Dict[str, list[float]]] = {}
    for name, spec in macros.items():
        in_dim = int(spec.get("in", 0))
        out_dim = int(spec.get("out", 0))
        weights = [[0.0 for _ in range(out_dim)] for _ in range(in_dim)]
        bias = [0.0 for _ in range(out_dim)]
        layers[name] = {"weights": weights, "bias": bias}
    return layers


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
