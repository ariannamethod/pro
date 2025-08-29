import json
import logging
import os
import threading
import hashlib
from typing import Dict, List

from pro_metrics import tokenize, compute_metrics
import pro_tune

STATE_PATH = 'pro_state.json'
HASH_PATH = 'dataset_sha.json'
LOG_PATH = 'pro.log'


class ProEngine:
    def __init__(self):
        self.state: Dict = {'word_counts': {}, 'bigram_counts': {}}
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH, 'r', encoding='utf-8') as fh:
                self.state = json.load(fh)
        if not self.state['word_counts']:
            # initial training
            pro_tune.train(self.state, 'datasets/lines01.txt')
            self.save_state()
        logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format='%(message)s')
        self.scan_datasets()

    def save_state(self) -> None:
        with open(STATE_PATH, 'w', encoding='utf-8') as fh:
            json.dump(self.state, fh)

    def scan_datasets(self) -> None:
        """Scan datasets directory and trigger async tuning on change."""
        if not os.path.exists('datasets'):
            return
        new_hashes = {}
        for name in os.listdir('datasets'):
            path = os.path.join('datasets', name)
            if not os.path.isfile(path):
                continue
            with open(path, 'rb') as fh:
                digest = hashlib.sha256(fh.read()).hexdigest()
            new_hashes[name] = digest
        old_hashes = {}
        if os.path.exists(HASH_PATH):
            with open(HASH_PATH, 'r', encoding='utf-8') as fh:
                old_hashes = json.load(fh)
        changed = any(old_hashes.get(k) != v for k, v in new_hashes.items())
        with open(HASH_PATH, 'w', encoding='utf-8') as fh:
            json.dump(new_hashes, fh)
        if changed:
            threading.Thread(target=self._async_tune, daemon=True).start()

    def _async_tune(self):
        for name in os.listdir('datasets'):
            path = os.path.join('datasets', name)
            pro_tune.train(self.state, path)
        self.save_state()

    def compute_charged_words(self, words: List[str]) -> List[str]:
        charges: Dict[str, float] = {}
        for w in words:
            freq = words.count(w)
            successors = len(self.state['bigram_counts'].get(w, {}))
            charges[w] = freq * (1 + successors)
        ordered = sorted(charges, key=charges.get, reverse=True)
        return ordered[:5]

    def respond(self, charged: List[str]) -> str:
        if not charged:
            return "Silence echoes within void."
        if len(charged) < 5:
            charged = (charged * 5)[:5]
        sentence = charged[0].capitalize() + ' ' + ' '.join(charged[1:]) + '.'
        return sentence

    def update_model(self, words: List[str]) -> None:
        wc = self.state.setdefault('word_counts', {})
        bc = self.state.setdefault('bigram_counts', {})
        prev = '<s>'
        wc[prev] = wc.get(prev, 0) + 1
        for word in words:
            wc[word] = wc.get(word, 0) + 1
            bc.setdefault(prev, {})
            bc[prev][word] = bc[prev].get(word, 0) + 1
            prev = word
        self.save_state()

    def log(self, user: str, response: str, metrics: Dict) -> None:
        logging.info(json.dumps({'user': user, 'response': response, 'metrics': metrics}))

    def analyze(self, message: str):
        words = tokenize(message)
        metrics = compute_metrics(words, self.state['bigram_counts'], self.state['word_counts'])
        charged = self.compute_charged_words(words)
        return metrics, charged

    def interact(self) -> None:
        while True:
            try:
                message = input('> ').strip()
            except EOFError:
                break
            if not message or message.lower() in {'exit', 'quit'}:
                break
            metrics, charged = self.analyze(message)
            response = self.respond(charged)
            print(response)
            self.log(message, response, metrics)
            self.update_model(tokenize(message))
            self.update_model(tokenize(response))


if __name__ == '__main__':
    engine = ProEngine()
    engine.interact()
