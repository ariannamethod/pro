import json
import logging
import os
import hashlib
import asyncio
from typing import Dict, List, Tuple
from collections import Counter

from pro_metrics import tokenize, compute_metrics, lowercase
import pro_tune
import pro_sequence
import pro_memory
import pro_rag

STATE_PATH = 'pro_state.json'
HASH_PATH = 'dataset_sha.json'
LOG_PATH = 'pro.log'


class ProEngine:
    def __init__(self) -> None:
        self.state: Dict = {'word_counts': {}, 'bigram_counts': {}}

    async def setup(self) -> None:
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH, 'r', encoding='utf-8') as fh:
                self.state = json.load(fh)
        if not self.state['word_counts']:
            try:
                await asyncio.to_thread(
                    pro_tune.train, self.state, 'datasets/lines01.txt'
                )  # noqa: E501
                await self.save_state()
            except Exception:
                pass
        await pro_memory.init_db()
        logging.basicConfig(
            filename=LOG_PATH, level=logging.INFO, format='%(message)s'
        )  # noqa: E501
        await self.scan_datasets()

    async def save_state(self) -> None:
        def _write():
            with open(STATE_PATH, 'w', encoding='utf-8') as fh:
                json.dump(self.state, fh)
        await asyncio.to_thread(_write)

    async def scan_datasets(self) -> None:
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
        changed = (
            set(old_hashes) != set(new_hashes)
            or any(old_hashes.get(k) != v for k, v in new_hashes.items())
        )
        with open(HASH_PATH, 'w', encoding='utf-8') as fh:
            json.dump(new_hashes, fh)
        if changed:
            asyncio.create_task(self._async_tune())

    async def _async_tune(self) -> None:
        for name in os.listdir('datasets'):
            path = os.path.join('datasets', name)
            await asyncio.to_thread(pro_tune.train, self.state, path)
        await self.save_state()

    def compute_charged_words(self, words: List[str]) -> List[str]:
        word_counts = Counter(words)
        charges: Dict[str, float] = {}
        for w, freq in word_counts.items():
            successors = len(self.state['bigram_counts'].get(w, {}))
            charges[w] = freq * (1 + successors)
        ordered = sorted(charges, key=charges.get, reverse=True)
        return ordered[:5]

    def respond(self, charged: List[str]) -> str:
        if not charged:
            return "Silence echoes within void."
        if len(charged) < 5:
            charged = (charged * 5)[:5]
        first = charged[0]
        if first and first[0].isalpha():
            first = first[0].upper() + first[1:]
        words = [first] + charged[1:]
        sentence = " ".join(filter(None, words)) + "."
        return sentence

    async def process_message(self, message: str) -> Tuple[str, Dict]:
        original_words = tokenize(message)
        words = lowercase(original_words)
        await pro_memory.add_message(message)
        context = await pro_rag.retrieve(words)
        context_tokens = tokenize(' '.join(context))
        all_words = words + lowercase(context_tokens)
        metrics = compute_metrics(
            all_words, self.state['bigram_counts'], self.state['word_counts']
        )
        charged = self.compute_charged_words(original_words + context_tokens)
        response = self.respond(charged)
        await pro_memory.add_message(response)
        await asyncio.to_thread(
            pro_sequence.analyze_sequences, self.state, words
        )
        await asyncio.to_thread(
            pro_sequence.analyze_sequences,
            self.state,
            lowercase(tokenize(response)),
        )
        await self.save_state()
        self.log(message, response, metrics)
        return response, metrics

    def log(self, user: str, response: str, metrics: Dict) -> None:
        logging.info(
            json.dumps(
                {
                    'user': user,
                    'response': response,
                    'metrics': metrics,
                }
            )
        )

    async def interact(self) -> None:
        await self.setup()
        while True:
            try:
                message = await asyncio.to_thread(input, '> ')
            except EOFError:
                break
            message = message.strip()
            if not message or message.lower() in {'exit', 'quit'}:
                break
            response, _ = await self.process_message(message)
            print(response)


if __name__ == '__main__':
    asyncio.run(ProEngine().interact())
