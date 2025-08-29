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
import pro_predict

STATE_PATH = 'pro_state.json'
HASH_PATH = 'dataset_sha.json'
LOG_PATH = 'pro.log'


class ProEngine:
    def __init__(self) -> None:
        self.state: Dict = {
            'word_counts': {},
            'bigram_counts': {},
            'trigram_counts': {},
            'char_ngram_counts': {},
        }

    async def setup(self) -> None:
        if os.path.exists(STATE_PATH):
            self.state = pro_tune.load_state(STATE_PATH)
        for key in [
            'word_counts',
            'bigram_counts',
            'trigram_counts',
            'char_ngram_counts',
        ]:
            self.state.setdefault(key, {})
        if not self.state['word_counts']:
            dataset_path = 'datasets/lines01.txt'
            if not os.path.exists(dataset_path):
                logging.warning(
                    "Dataset path %s does not exist; skipping initial training",
                    dataset_path,
                )
            elif os.path.getsize(dataset_path) == 0:
                logging.warning(
                    "Dataset path %s is empty; skipping initial training",
                    dataset_path,
                )
            else:
                try:
                    await asyncio.to_thread(pro_tune.train, self.state, dataset_path)
                    await self.save_state()
                    logging.info("Initial training succeeded on %s", dataset_path)
                except Exception as exc:
                    logging.error(
                        "Initial training failed: %s", exc
                    )  # pragma: no cover - logging side effect
        await pro_memory.init_db()
        logging.basicConfig(
            filename=LOG_PATH, level=logging.INFO, format='%(message)s'
        )  # noqa: E501
        await self.scan_datasets()
        asyncio.create_task(self._dataset_watcher())

    async def save_state(self) -> None:
        await asyncio.to_thread(pro_tune.save_state, self.state, STATE_PATH)

    async def scan_datasets(self) -> None:
        if not os.path.exists('datasets'):
            return
        old_hashes: Dict[str, str] = {}
        if os.path.exists(HASH_PATH):
            with open(HASH_PATH, 'r', encoding='utf-8') as fh:
                old_hashes = json.load(fh)
        new_hashes: Dict[str, str] = {}
        changed_files: List[str] = []
        for name in os.listdir('datasets'):
            path = os.path.join('datasets', name)
            if not os.path.isfile(path):
                continue
            with open(path, 'rb') as fh:
                digest = hashlib.sha256(fh.read()).hexdigest()
            new_hashes[name] = digest
            if old_hashes.get(name) != digest:
                changed_files.append(path)
        removed = set(old_hashes) - set(new_hashes)
        if removed and not new_hashes:
            with open(HASH_PATH, 'w', encoding='utf-8') as fh:
                json.dump(new_hashes, fh)
            asyncio.create_task(self._async_tune([]))
            return
        if removed:
            changed_files = [
                os.path.join('datasets', n) for n in new_hashes.keys()
            ]
        with open(HASH_PATH, 'w', encoding='utf-8') as fh:
            json.dump(new_hashes, fh)
        if changed_files:
            asyncio.create_task(self._async_tune(changed_files))

    async def _async_tune(self, paths: List[str]) -> None:
        tuned: List[str] = []
        for path in paths:
            try:
                await asyncio.to_thread(pro_tune.train, self.state, path)
                tuned.append(os.path.basename(path))
            except Exception as exc:  # pragma: no cover - logging side effect
                logging.error("Tuning failed for %s: %s", path, exc)
        try:
            await self.save_state()
        except Exception as exc:  # pragma: no cover - logging side effect
            logging.error("Saving state failed after tuning: %s", exc)
        if tuned:
            logging.info("Tuned datasets: %s", ", ".join(tuned))

    async def _dataset_watcher(self) -> None:
        while True:
            try:
                await self.scan_datasets()
            except Exception as exc:  # pragma: no cover - logging side effect
                logging.error("Dataset scan failed: %s", exc)
            await asyncio.sleep(60)

    def compute_charged_words(self, words: List[str]) -> List[str]:
        word_counts = Counter(words)
        charges: Dict[str, float] = {}
        for w, freq in word_counts.items():
            successors = len(self.state['bigram_counts'].get(w, {}))
            charges[w] = freq * (1 + successors)
        ordered = sorted(charges, key=charges.get, reverse=True)
        return ordered[:5]

    def predict_next_word(self, prev2: str, prev1: str) -> str:
        tc = self.state.get("trigram_counts", {})
        bc = self.state.get("bigram_counts", {})
        wc = self.state.get("word_counts", {})
        candidates = tc.get((prev2, prev1))
        if candidates:
            return max(candidates, key=candidates.get)
        candidates2 = bc.get(prev1)
        if candidates2:
            return max(candidates2, key=candidates2.get)
        if wc:
            return max(wc, key=wc.get)
        return ""

    def respond(self, seeds: List[str]) -> str:
        if not seeds:
            return "Silence echoes within void."
        words = [w for w in seeds if w]
        word_counts = self.state.get("word_counts", {})
        ordered = sorted(word_counts, key=word_counts.get, reverse=True)
        for w in ordered:
            if len(words) >= 2:
                break
            if w and w not in words:
                words.append(w)
        while len(words) < 2:
            words.append("")
        while len(words) < 5:
            prev2, prev1 = words[-2], words[-1]
            nxt = self.predict_next_word(prev2, prev1)
            if not nxt or nxt in words:
                fallback = next(
                    (w for w in ordered if w and w not in words),
                    None,
                )
                if fallback is not None:
                    nxt = fallback
                else:
                    nxt = words[(len(words)) % len(words)]
            words.append(nxt)
        first = words[0]
        if first and first[0].isalpha():
            first = first[0].upper() + first[1:]
        words[0] = first
        sentence = " ".join(filter(None, words[:5])) + "."
        return sentence

    async def process_message(self, message: str) -> Tuple[str, Dict]:
        original_words = tokenize(message)
        words = lowercase(original_words)
        unknown: List[str] = [
            w for w in words if w not in self.state['word_counts']
        ]
        predicted: List[str] = []
        for w in unknown:
            predicted.extend(pro_predict.suggest(w))
        try:
            await pro_memory.add_message(message)
        except Exception as exc:  # pragma: no cover - logging side effect
            logging.error("Storing message failed: %s", exc)
        context: List[str] = []
        try:
            context = await pro_rag.retrieve(words)
        except Exception as exc:  # pragma: no cover - logging side effect
            logging.error("Context retrieval failed: %s", exc)
        context_tokens = tokenize(' '.join(context))
        all_words = words + lowercase(context_tokens)
        metrics = compute_metrics(
            all_words,
            self.state['trigram_counts'],
            self.state['bigram_counts'],
            self.state['word_counts'],
            self.state['char_ngram_counts'],
        )
        seed_words = original_words + context_tokens + predicted
        response = self.respond(seed_words)
        try:
            await pro_memory.add_message(response)
        except Exception as exc:  # pragma: no cover - logging side effect
            logging.error("Storing response failed: %s", exc)
        await asyncio.to_thread(
            pro_sequence.analyze_sequences, self.state, words
        )
        await asyncio.to_thread(
            pro_sequence.analyze_sequences,
            self.state,
            lowercase(tokenize(response)),
        )
        try:
            await self.save_state()
        except Exception as exc:  # pragma: no cover - logging side effect
            logging.error("Saving state failed: %s", exc)
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
            try:
                response, _ = await self.process_message(message)
            except Exception as exc:  # pragma: no cover - logging side effect
                logging.error("Processing message failed: %s", exc)
                print("An error occurred. Please try again.")
                continue
            print(response)


if __name__ == '__main__':
    asyncio.run(ProEngine().interact())
