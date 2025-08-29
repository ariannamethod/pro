import json
import logging
import os
import hashlib
import asyncio
import math
from typing import Dict, List, Tuple, Set, Optional
from collections import Counter

from pro_metrics import (
    tokenize,
    compute_metrics,
    lowercase,
    target_length_from_metrics,
)
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
                    "Dataset path %s does not exist; "
                    "skipping initial training",
                    dataset_path,
                )
            elif os.path.getsize(dataset_path) == 0:
                logging.warning(
                    "Dataset path %s is empty; skipping initial training",
                    dataset_path,
                )
            else:
                try:
                    await asyncio.to_thread(
                        pro_tune.train, self.state, dataset_path
                    )
                    await self.save_state()
                    logging.info(
                        "Initial training succeeded on %s", dataset_path
                    )
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

    def plan_sentence(
        self,
        initial: List[str],
        target_length: int,
        beam_width: int = 3,
        forbidden: Optional[Set[str]] = None,
    ) -> List[str]:
        """Plan a sentence using beam search.

        Candidates are scored by combining entropy and trigram perplexity.
        Paths that would repeat words (case-insensitive) are discarded.
        """
        trigram_counts = self.state.get("trigram_counts", {})
        bigram_counts = self.state.get("bigram_counts", {})
        word_counts = self.state.get("word_counts", {})
        char_counts = self.state.get("char_ngram_counts", {})

        base_used: Set[str] = set(w.lower() for w in forbidden or [])
        start_seq: List[str] = []
        for w in initial:
            lw = w.lower()
            if w and lw not in base_used:
                start_seq.append(w)
                base_used.add(lw)

        global_order = sorted(word_counts, key=word_counts.get, reverse=True)
        beams = [(start_seq, base_used)]
        while beams and len(beams[0][0]) < target_length:
            new_beams = []
            for seq, used in beams:
                if len(seq) >= 2:
                    prev2, prev1 = seq[-2], seq[-1]
                elif len(seq) == 1:
                    prev2, prev1 = "<s>", seq[-1]
                else:
                    prev2, prev1 = "<s>", "<s>"
                tcands = trigram_counts.get((prev2, prev1), {})
                ordered = sorted(tcands, key=tcands.get, reverse=True)
                if ordered:
                    fallback = [w for w in global_order if w not in ordered]
                    ordered.extend(fallback)
                else:
                    ordered = [w for w in global_order if w.lower() not in used]
                if not ordered:
                    ordered = [f"alt{len(used)+i}" for i in range(beam_width * 2)]
                for cand in ordered[: beam_width * 2]:
                    lw = cand.lower()
                    if lw in used:
                        continue
                    new_seq = seq + [cand]
                    new_used = used | {lw}
                    metrics = compute_metrics(
                        [w.lower() for w in new_seq],
                        trigram_counts,
                        bigram_counts,
                        word_counts,
                        char_counts,
                    )
                    score = metrics["entropy"] - metrics["trigram_perplexity"]
                    new_beams.append((score, new_seq, new_used))
            if not new_beams:
                break
            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = [(seq, used) for _, seq, used in new_beams[:beam_width]]
        best_seq = beams[0][0] if beams else start_seq
        return best_seq[:target_length]

    def respond(self, seeds: List[str]) -> str:
        if not seeds:
            return "Silence echoes within void."

        dataset_words: List[str] = []
        if os.path.exists("datasets"):
            for name in os.listdir("datasets"):
                path = os.path.join("datasets", name)
                if not os.path.isfile(path):
                    continue
                try:
                    with open(
                        path, "r", encoding="utf-8", errors="ignore"
                    ) as fh:
                        for line in fh:
                            dataset_words.extend(
                                [w for w in line.strip().split() if w]
                            )
                except Exception:  # pragma: no cover - safety
                    continue

        attempt_seeds = list(seeds)
        extra_idx = 0
        while True:
            # ----- First sentence -----
            words: List[str] = []
            used = set()
            for w in attempt_seeds:
                if w and w not in used:
                    words.append(w)
                    used.add(w)
            word_counts = self.state.get("word_counts", {})
            ordered = sorted(word_counts, key=word_counts.get, reverse=True)
            for w in ordered:
                if len(words) >= 2:
                    break
                if w and w not in used:
                    words.append(w)
                    used.add(w)
            while len(words) < 2:
                words.append("")
            metrics = compute_metrics(
                [w.lower() for w in words if w],
                self.state.get("trigram_counts", {}),
                self.state.get("bigram_counts", {}),
                self.state.get("word_counts", {}),
                self.state.get("char_ngram_counts", {}),
            )
            target_length = target_length_from_metrics(metrics)
            words = self.plan_sentence(words, target_length)
            first = words[0]
            if first and first[0].isalpha():
                first = first[0].upper() + first[1:]
            words[0] = first
            sentence1 = " ".join(filter(None, words[:target_length])) + "."
            first_words = lowercase(tokenize(sentence1))
            used = set(first_words)

            # ----- Second sentence: choose semantically distant seeds -----
            pro_predict._ensure_vectors()
            vectors = pro_predict._VECTORS

            def _cos(a: Dict[str, float], b: Dict[str, float]) -> float:
                keys = set(a) | set(b)
                dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
                na = math.sqrt(sum(v * v for v in a.values()))
                nb = math.sqrt(sum(v * v for v in b.values()))
                if na == 0 or nb == 0:
                    return 0.0
                return dot / (na * nb)

            first_vecs = [vectors[w] for w in first_words if w in vectors]
            scores: Dict[str, float] = {}
            for word, vec in vectors.items():
                if word in used:
                    continue
                if first_vecs:
                    sim = max(_cos(vec, fv) for fv in first_vecs)
                else:
                    sim = 0.0
                scores[word] = sim
            ordered2 = (
                [w for w, _ in sorted(scores.items(), key=lambda x: x[1])]
                if scores
                else [w for w in ordered if w not in used]
            )
            second_seeds = ordered2[:2]

            metrics_first = compute_metrics(
                first_words,
                self.state.get("trigram_counts", {}),
                self.state.get("bigram_counts", {}),
                self.state.get("word_counts", {}),
                self.state.get("char_ngram_counts", {}),
            )
            target_length2 = target_length_from_metrics(
                {
                    "entropy": metrics_first["entropy"],
                    "perplexity": metrics_first["perplexity"],
                },
                min_len=5,
                max_len=10,
            )
            words2 = self.plan_sentence(
                second_seeds + ordered,
                target_length2,
                forbidden=set(first_words),
            )
            first2 = words2[0]
            if first2 and first2[0].isalpha():
                first2 = first2[0].upper() + first2[1:]
            words2[0] = first2
            sentence2 = " ".join(filter(None, words2[:target_length2])) + "."

            response = sentence1 + " " + sentence2
            if pro_memory.is_unique(response):
                pro_memory.store_response(response)
                return response
            if extra_idx < len(dataset_words):
                attempt_seeds = list(seeds) + [dataset_words[extra_idx]]
            else:
                attempt_seeds = list(seeds) + [f"alt{extra_idx}"]
            extra_idx += 1

    async def process_message(self, message: str) -> Tuple[str, Dict]:
        original_words = tokenize(message)
        words = lowercase(original_words)
        unknown: List[str] = [
            w for w in words if w not in self.state['word_counts']
        ]
        predicted: List[str] = []
        for w in unknown:
            predicted.extend(pro_predict.suggest(w))
        # Blend n-gram prediction with transformer logits
        ngram_pred = ""
        if words:
            prev2 = words[-2] if len(words) >= 2 else ""
            prev1 = words[-1]
            ngram_pred = self.predict_next_word(prev2, prev1)
        trans_pred = ""
        vocab = list(self.state.get("word_counts", {}).keys())
        if vocab:
            logits = pro_predict.transformer_logits(words[-5:], vocab)
            trans_pred = max(logits, key=logits.get)
        blend: List[str] = []
        if ngram_pred:
            blend.append(ngram_pred)
        if trans_pred and trans_pred != ngram_pred:
            blend.append(trans_pred)
        predicted.extend(blend)
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
        await pro_predict.update(words + lowercase(tokenize(response)))
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
