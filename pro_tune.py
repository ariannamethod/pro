import argparse
import json
import logging
import os
from typing import Dict

from pro_metrics import tokenize, lowercase
import pro_sequence

STATE_PATH = 'pro_state.json'
_SEP = '\u0001'


def train(state: Dict, dataset_path: str) -> Dict:
    if not os.path.exists(dataset_path):
        logging.warning("Dataset path %s does not exist; skipping training", dataset_path)
        return state
    with open(dataset_path, 'r', encoding='utf-8') as fh:
        text = fh.read()
    if not text:
        logging.warning("Dataset path %s is empty; skipping training", dataset_path)
        return state
    words = lowercase(tokenize(text))
    pro_sequence.analyze_sequences(state, words)
    return state


def _serialize_state(state: Dict) -> Dict:
    data = dict(state)
    tc = {
        f"{k[0]}{_SEP}{k[1]}": v
        for k, v in state.get('trigram_counts', {}).items()
    }
    data['trigram_counts'] = tc
    return data


def _deserialize_state(state: Dict) -> Dict:
    tc = {}
    for k, v in state.get('trigram_counts', {}).items():
        parts = k.split(_SEP)
        if len(parts) == 2:
            tc[(parts[0], parts[1])] = v
    state['trigram_counts'] = tc
    return state


def save_state(state: Dict, path: str = STATE_PATH) -> None:
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(_serialize_state(state), fh)


def load_state(path: str = STATE_PATH) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    return _deserialize_state(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train model on a dataset")
    parser.add_argument("dataset_path", help="Path to dataset for training")
    parser.add_argument(
        "--state-path", default=STATE_PATH, help="Path to state file"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    state = load_state(args.state_path)
    state = train(state, args.dataset_path)
    save_state(state, args.state_path)
    logging.info("Training complete for %s", args.dataset_path)
