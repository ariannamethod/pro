import json
import logging
import os
import hashlib
import asyncio
import math
import random
import re
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
from pro_identity import swap_pronouns
from watchfiles import awatch

STATE_PATH = 'pro_state.json'
HASH_PATH = 'dataset_sha.json'
LOG_PATH = 'pro.log'


class ProEngine:
    def __init__(
        self, chaos_factor: float = 0.0, similarity_threshold: float = 0.3
    ) -> None:
        self.state: Dict = {
            'word_counts': {},
            'bigram_counts': {},
            'trigram_counts': {},
            'char_ngram_counts': {},
            'word_inv': {},
            'bigram_inv': {},
            'trigram_inv': {},
            'char_ngram_inv': {},
        }
        self.chaos_factor = chaos_factor
        self.similarity_threshold = similarity_threshold

    async def setup(self) -> None:
        pro_predict._GRAPH = {}
        pro_predict._VECTORS = {}
        if os.path.exists(STATE_PATH):
            self.state = pro_tune.load_state(STATE_PATH)
        for key in [
            'word_counts',
            'bigram_counts',
            'trigram_counts',
            'char_ngram_counts',
            'word_inv',
            'bigram_inv',
            'trigram_inv',
            'char_ngram_inv',
        ]:
            self.state.setdefault(key, {})
        # Recompute inverse-frequency maps from counts
        for w, c in self.state['word_counts'].items():
            self.state['word_inv'][w] = 1.0 / c
        for prev, foll in self.state['bigram_counts'].items():
            bi = self.state['bigram_inv'].setdefault(prev, {})
            for w, c in foll.items():
                bi[w] = 1.0 / c
        for key, foll in self.state['trigram_counts'].items():
            ti = self.state['trigram_inv'].setdefault(key, {})
            for w, c in foll.items():
                ti[w] = 1.0 / c
        for ngram, c in self.state['char_ngram_counts'].items():
            self.state['char_ngram_inv'][ngram] = 1.0 / c
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
        os.makedirs('datasets', exist_ok=True)
        await self.scan_datasets()

        async def _watch_datasets() -> None:
            async for _ in awatch('datasets'):
                try:
                    await self.scan_datasets()
                except Exception as exc:  # pragma: no cover - logging side effect
                    logging.error("Dataset scan failed: %s", exc)

        asyncio.create_task(_watch_datasets())

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
                new_state = await asyncio.to_thread(
                    pro_tune.train, self.state, path
                )
                if new_state is not None:
                    self.state = new_state
                tuned.append(os.path.basename(path))
            except Exception as exc:  # pragma: no cover - logging side effect
                logging.error("Tuning failed for %s: %s", path, exc)
        try:
            await self.save_state()
        except Exception as exc:  # pragma: no cover - logging side effect
            logging.error("Saving state failed after tuning: %s", exc)
        if tuned:
            logging.info("Tuned datasets: %s", ", ".join(tuned))


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
        chaos_factor: float = 0.0,
    ) -> List[str]:
        """Plan a sentence using beam search.

        Candidates are scored by combining entropy and trigram perplexity.
        Paths that would repeat words (case-insensitive) are discarded.
        """
        trigram_counts = self.state.get("trigram_counts", {})
        bigram_counts = self.state.get("bigram_counts", {})
        word_counts = self.state.get("word_counts", {})
        char_counts = self.state.get("char_ngram_counts", {})
        trigram_inv = self.state.get("trigram_inv", {})
        bigram_inv = self.state.get("bigram_inv", {})
        word_inv = self.state.get("word_inv", {})

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
                    ordered = [
                        w for w in global_order if w.lower() not in used
                    ]
                if not ordered:
                    ordered = [
                        f"alt{len(used)+i}" for i in range(beam_width * 2)
                    ]
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
                    if chaos_factor:
                        inv = trigram_inv.get((prev2, prev1), {}).get(
                            cand,
                            bigram_inv.get(prev1, {}).get(
                                cand, word_inv.get(cand, 0.0)
                            ),
                        )
                        score += chaos_factor * inv * random.random()
                    new_beams.append((score, new_seq, new_used))
            if not new_beams:
                break
            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = [(seq, used) for _, seq, used in new_beams[:beam_width]]
        best_seq = beams[0][0] if beams else start_seq
        return best_seq[:target_length]

    async def respond(
        self,
        seeds: List[str],
        vocab: Optional[Dict[str, int]] = None,
        chaos_factor: Optional[float] = None,
        similarity_threshold: Optional[float] = None,
        forbidden: Optional[Set[str]] = None,
    ) -> str:
        if not seeds:
            return "Silence echoes within void."

        vocab = vocab or {}
        cf = self.chaos_factor if chaos_factor is None else chaos_factor
        forbidden = {w.lower() for w in (forbidden or set())}
        analog_map: Dict[str, str] = {}
        for tok in forbidden:
            suggestions = pro_predict.suggest(tok, topn=1)
            analog = suggestions[0] if suggestions else None
            if not analog:
                analog = pro_predict.lookup_analogs(tok)
            if analog:
                analog_map[tok] = analog
        ordered_vocab: List[str] = []
        seen_vocab: Set[str] = set()
        for w, _ in sorted(vocab.items(), key=lambda x: x[1], reverse=True):
            lw = w.lower()
            if lw in forbidden and lw not in analog_map:
                continue
            repl = analog_map.get(lw, w)
            if repl not in seen_vocab:
                ordered_vocab.append(repl)
                seen_vocab.add(repl)

        attempt_seeds: List[str] = []
        for w in seeds:
            lw = w.lower()
            repl = analog_map.get(lw, w)
            if w.isupper():
                repl = repl.upper()
            elif w and w[0].isupper():
                repl = repl[0].upper() + repl[1:]
            attempt_seeds.append(repl)
        extra_idx = 0
        while True:
            # ----- First sentence -----
            metrics = compute_metrics(
                [w.lower() for w in attempt_seeds if w],
                self.state.get("trigram_counts", {}),
                self.state.get("bigram_counts", {}),
                self.state.get("word_counts", {}),
                self.state.get("char_ngram_counts", {}),
            )
            bigram_inv = self.state.get("bigram_inv", {})
            trigram_inv = self.state.get("trigram_inv", {})
            inv_scores: Dict[str, float] = {}
            for w in attempt_seeds:
                lw = w.lower()
                bi_val = max(
                    (bigram_inv.get(p, {}).get(lw, 0.0) for p in bigram_inv),
                    default=0.0,
                )
                tri_val = max(
                    (trigram_inv.get(k, {}).get(lw, 0.0) for k in trigram_inv),
                    default=0.0,
                )
                inv_scores[lw] = max(bi_val, tri_val)
            max_inv = max(inv_scores.values(), default=0.0)
            high_inv_words = {
                w for w, v in inv_scores.items() if v == max_inv and v > 0.0
            }

            target_length = target_length_from_metrics(metrics)
            words: List[str] = []
            tracker: Set[str] = set()
            for w in attempt_seeds:
                analog = pro_predict.lookup_analogs(w.lower()) or w
                if w.isupper():
                    analog = analog.upper()
                elif w and w[0].isupper():
                    analog = analog[0].upper() + analog[1:]
                if analog and analog.lower() not in forbidden and analog not in tracker:
                    words.append(analog)
                    tracker.add(analog)
            word_counts = self.state.get("word_counts", {})
            combined_counts: Dict[str, float] = dict(word_counts)
            for w, weight in vocab.items():
                lw = w.lower()
                if lw in forbidden and lw not in analog_map:
                    continue
                w = analog_map.get(lw, w)
                combined_counts[w] = combined_counts.get(w, 0) + weight
            for w in high_inv_words:
                analog = pro_predict.lookup_analogs(w)
                if analog:
                    combined_counts[analog] = (
                        combined_counts.get(analog, 0.0)
                        + combined_counts.get(w, 0.0)
                        + 1.0
                    )
                    combined_counts.pop(w, None)
                    tracker.discard(w)
            ordered = sorted(
                combined_counts, key=combined_counts.get, reverse=True
            )
            for w in ordered:
                if len(words) >= 2:
                    break
                if w and w.lower() not in forbidden and w not in tracker:
                    words.append(w)
                    tracker.add(w)
            while len(words) < 2:
                words.append("")
            words = self.plan_sentence(
                words,
                target_length,
                forbidden=forbidden,
                chaos_factor=cf,
            )
            first = words[0]
            if first and first[0].isalpha():
                first = first[0].upper() + first[1:]
            words[0] = first
            sentence1 = " ".join(filter(None, words[:target_length])) + "."
            first_words = lowercase(tokenize(sentence1))
            tracker = set(first_words)

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
                if word in tracker:
                    continue
                if first_vecs:
                    sim = max(_cos(vec, fv) for fv in first_vecs)
                else:
                    sim = 0.0
                scores[word] = sim
            sim_thresh = (
                self.similarity_threshold
                if similarity_threshold is None
                else similarity_threshold
            )
            eligible = [w for w, s in scores.items() if s < sim_thresh]
            if eligible:
                ordered2 = sorted(eligible, key=lambda w: scores[w])
            else:
                ordered2 = (
                    [w for w, _ in sorted(scores.items(), key=lambda x: x[1])]
                    if scores
                    else [w for w in ordered if w not in tracker]
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
                max_len=6,
            )
            words2 = self.plan_sentence(
                second_seeds + ordered,
                target_length2,
                forbidden=set(first_words) | forbidden,
                chaos_factor=cf,
            )
            first2 = words2[0]
            if first2 and first2[0].isalpha():
                first2 = first2[0].upper() + first2[1:]
            words2[0] = first2
            sentence2 = " ".join(filter(None, words2[:target_length2])) + "."
            last1 = sentence1.rstrip(".").split()[-1].lower()
            if last1 in {"the", "a", "and", "or"}:
                replacement = next(
                    (
                        w
                        for w in ordered
                        if w.lower() not in {"the", "a", "and", "or"}
                    ),
                    last1,
                )
                parts = sentence1.rstrip(".").split()
                parts[-1] = replacement
                sentence1 = " ".join(parts) + "."

            last2 = sentence2.rstrip(".").split()[-1].lower()
            if last2 in {"the", "a", "and", "or"}:
                replacement2 = next(
                    (
                        w
                        for w in ordered2
                        if w.lower() not in {"the", "a", "and", "or"}
                    ),
                    last2,
                )
                parts2 = sentence2.rstrip(".").split()
                parts2[-1] = replacement2
                sentence2 = " ".join(parts2) + "."

            response = sentence1 + " " + sentence2
            for tok, analog in analog_map.items():
                pattern = re.compile(rf"\b{re.escape(tok)}\b", re.IGNORECASE)
                response = pattern.sub(analog, response)
            if await pro_memory.is_unique(response):
                await pro_memory.store_response(response)
                return response
            if extra_idx < len(ordered_vocab):
                attempt_seeds = list(attempt_seeds) + [ordered_vocab[extra_idx]]
            else:
                attempt_seeds = list(seeds) + [f"alt{extra_idx}"]
            extra_idx += 1

    async def process_message(self, message: str) -> Tuple[str, Dict]:
        original_words = tokenize(message)
        words = lowercase(original_words)
        words = swap_pronouns(words)
        user_forbidden = set(words)
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
        recent_msgs, recent_resps = await pro_memory.fetch_recent(50)
        mem_tokens: List[str] = []
        for text in recent_msgs + recent_resps:
            mem_tokens.extend(lowercase(tokenize(text)))
        data_tokens: List[str] = []
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
                            data_tokens.extend(lowercase(tokenize(line)))
                except Exception:  # pragma: no cover - safety
                    continue
        mem_counts = Counter(mem_tokens)
        data_counts = Counter(data_tokens)
        combined_vocab: Dict[str, int] = {}
        for w in set(mem_counts) | set(data_counts):
            weight = mem_counts.get(w, 0) + data_counts.get(w, 0)
            if w in mem_counts and w in data_counts:
                weight *= 2
            combined_vocab[w] = weight
        if self.chaos_factor:
            response = await self.respond(
                seed_words,
                combined_vocab,
                chaos_factor=self.chaos_factor,
                forbidden=user_forbidden,
            )
        else:
            response = await self.respond(
                seed_words, combined_vocab, forbidden=user_forbidden
            )
        try:
            await pro_memory.add_message(response)
        except Exception as exc:  # pragma: no cover - logging side effect
            logging.error("Storing response failed: %s", exc)
        dataset_path = os.path.join('datasets', 'conversation.log')
        try:
            os.makedirs('datasets', exist_ok=True)
            with open(dataset_path, 'a', encoding='utf-8') as fh:
                fh.write(f"{message}\n{response}\n")
            try:
                with open(dataset_path, 'rb') as fh:
                    digest = hashlib.sha256(fh.read()).hexdigest()
                hashes = {}
                if os.path.exists(HASH_PATH):
                    with open(HASH_PATH, 'r', encoding='utf-8') as fh:
                        hashes = json.load(fh)
                hashes[os.path.basename(dataset_path)] = digest
                with open(HASH_PATH, 'w', encoding='utf-8') as fh:
                    json.dump(hashes, fh)
            except Exception as exc:  # pragma: no cover - logging side effect
                logging.error("Updating dataset hash failed: %s", exc)
            asyncio.create_task(self._async_tune([dataset_path]))
        except Exception as exc:  # pragma: no cover - logging side effect
            logging.error("Logging conversation failed: %s", exc)
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
