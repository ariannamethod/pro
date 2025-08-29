import json
import os
from typing import Dict

from pro_metrics import tokenize, lowercase

STATE_PATH = 'pro_state.json'


def train(state: Dict, dataset_path: str) -> Dict:
    if not os.path.exists(dataset_path):
        return state
    with open(dataset_path, 'r', encoding='utf-8') as fh:
        text = fh.read()
    words = lowercase(tokenize(text))
    wc = state.setdefault('word_counts', {})
    bc = state.setdefault('bigram_counts', {})
    prev = '<s>'
    wc[prev] = wc.get(prev, 0) + 1
    for word in words:
        wc[word] = wc.get(word, 0) + 1
        bc.setdefault(prev, {})
        bc[prev][word] = bc[prev].get(word, 0) + 1
        prev = word
    return state


def save_state(state: Dict, path: str = STATE_PATH) -> None:
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(state, fh)


if __name__ == '__main__':
    state = {}
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, 'r', encoding='utf-8') as fh:
            state = json.load(fh)
    state = train(state, 'datasets/lines01.txt')
    save_state(state)
