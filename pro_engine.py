import json
import logging
import os
import hashlib
import asyncio
import subprocess
import shutil
import math
import random
import re
import importlib.util
import sys
import time
from typing import Dict, List, Tuple, Set, Optional
from collections import Counter, deque

import numpy as np

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
import pro_rag_embedding
import pro_predict
import pro_forecast
import pro_meta
import dream_mode
import lora_utils
from autoadapt import LayerMutator
from pro_identity import swap_pronouns
import grammar_filters
from watchfiles import awatch
from transformers.blocks import SymbolicReasoner, LightweightMoEBlock
from meta_controller import MetaController
from api import vector_store
import pro_spawn
from metrics.timing import timed

STATE_PATH = 'pro_state.json'
HASH_PATH = 'dataset_sha.json'
LOG_PATH = 'pro.log'
TUNE_CONCURRENCY = 4
SCAN_CONCURRENCY = 4
COMPRESSION_INTERVAL = 100

FORBIDDEN_ENDINGS = {"the", "a", "and", "or", "his", "my", "their"}


class ProEngine:
    def __init__(
        self,
        chaos_factor: float = 0.0,
        similarity_threshold: float = 0.3,
        saliency_threshold: float = 0.0,
        novelty_threshold: float = 0.9,
        dream_interval: float = 0.5,
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
        self.saliency_threshold = saliency_threshold
        self.novelty_threshold = novelty_threshold
        self.candidate_buffer: deque = deque(maxlen=20)
        self.last_forecast: Optional[Dict] = None
        self.dataset_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self._tune_worker_task: Optional[asyncio.Task] = None
        self._watcher_task: Optional[asyncio.Task] = None
        self._tune_tasks: List[asyncio.Task] = []
        self._running_tasks: List[asyncio.Task] = []
        self._tune_semaphore = asyncio.BoundedSemaphore(TUNE_CONCURRENCY)
        self._compression_task: Optional[asyncio.Task] = None
        self._dream_task: Optional[asyncio.Task] = None
        self._dream_event = asyncio.Event()
        self._dream_interval = dream_interval
        self.adapter_pool = self._load_adapters()
        self.reasoner = SymbolicReasoner()
        self.light_moe = LightweightMoEBlock(dim=16, num_experts=4)
        self.meta_controller = MetaController(self)
        self.layer_config: Dict[str, int] = {}

    def _load_adapters(self) -> Dict[str, Dict]:
        pool: Dict[str, Dict] = {}
        base = "adapter_pool"
        if not os.path.isdir(base):
            return pool
        for name in os.listdir(base):
            cfg_path = os.path.join(base, name, "config.json")
            if not os.path.isfile(cfg_path):
                continue
            try:
                with open(cfg_path, "r", encoding="utf-8") as fh:
                    cfg = json.load(fh)
                weights_file = cfg.get("weights_path", "weights.json")
                wpath = os.path.join(base, name, weights_file)
                weights: Dict[str, float] = {}
                if os.path.isfile(wpath):
                    with open(wpath, "r", encoding="utf-8") as fh:
                        weights = json.load(fh)
                pool[name] = {"config": cfg, "weights": weights}
            except Exception:
                continue
        return pool

    def select_adapters(
        self, prompt: str, top_k: int = 1
    ) -> List[Tuple[str, Dict[str, float]]]:
        tokens = lowercase(tokenize(prompt))
        scores: List[Tuple[int, str]] = []
        for name, data in self.adapter_pool.items():
            keywords = data.get("config", {}).get("keywords", [])
            score = sum(tokens.count(k) for k in keywords)
            if score:
                scores.append((score, name))
        scores.sort(reverse=True)
        return [
            (n, self.adapter_pool[n]["weights"])
            for _, n in scores[:top_k]
        ]

    # Lightweight MoE -----------------------------------------------------

    @timed(name="light_moe_decode")
    def light_moe_decode(
        self,
        x: np.ndarray,
        adapters: Optional[List[Dict[str, float]]] = None,
    ) -> np.ndarray:
        """Decode vector *x* using the lightweight MoE block.

        Parameters
        ----------
        x:
            Input vector of shape ``(16,)``.
        adapters:
            Optional list of adapter weight dictionaries returned by
            :meth:`select_adapters`.
        """

        if self.light_moe is None:
            return x.astype(np.float32)
        return self.light_moe(x.astype(np.float32), adapters=adapters)

    # Dynamic module loading --------------------------------------------

    def load_generated_block(self, code: str, name: str = "GeneratedBlock"):
        """Compile and register a generated block module.

        The provided ``code`` should define a class with the given ``name``.
        The resulting class is exposed via :mod:`transformers.blocks` so it can
        be instantiated by training routines.

        Parameters
        ----------
        code:
            Source code of the module.
        name:
            Name of the class defined in ``code``.

        Returns
        -------
        types.ModuleType
            The compiled module object.
        """

        module_name = f"_generated_{name.lower()}"
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        module = importlib.util.module_from_spec(spec)
        exec(compile(code, module_name, "exec"), module.__dict__)
        sys.modules[module_name] = module

        try:
            import transformers.blocks as tblocks

            block_cls = getattr(module, name)
            setattr(tblocks, name, block_cls)
        except Exception:
            pass

        return module

    # Saliency helpers -------------------------------------------------

    def score_tokens(self, tokens: List[str]) -> List[float]:
        """Return an importance score for each token.

        Tokens are scored using inverse word frequency from the current
        engine state. Unknown tokens receive a default score of ``1.0``.
        """

        counts = self.state.get("word_counts", {})
        scores: List[float] = []
        for tok in tokens:
            c = counts.get(tok.lower(), 0)
            scores.append(1.0 / c if c > 0 else 1.0)
        return scores

    def _drop_low_saliency(self, tokens: List[str]) -> List[str]:
        """Filter out tokens below the configured saliency percentile."""

        if self.saliency_threshold <= 0.0 or not tokens:
            return tokens
        scores = self.score_tokens(tokens)
        cutoff = float(np.percentile(scores, self.saliency_threshold))
        return [tok for tok, score in zip(tokens, scores) if score >= cutoff]

    def _apply_layer_config(self, cfg: Dict[str, int]) -> None:
        """Apply the selected layer configuration to the reasoner and MoE."""

        self.layer_config = cfg
        if hasattr(self.reasoner, "configure"):
            try:
                self.reasoner.configure(cfg)
            except Exception:
                pass

        if cfg.get("use_light_moe", 1):
            dim = cfg.get("light_moe_dim", 16)
            num_experts = cfg.get("light_moe_experts", 4)
            top_k = cfg.get("light_moe_topk", 1)
            self.light_moe = LightweightMoEBlock(
                dim=dim, num_experts=num_experts, top_k=top_k
            )
        else:
            self.light_moe = None

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
        self.state.setdefault('architectures', [])
        arch_cfg = self.meta_controller.select()
        self._apply_layer_config(arch_cfg)
        logging.info("Selected architecture: %s", arch_cfg)
        self.state['architectures'].append(arch_cfg)
        await self.save_state()
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
            dataset_path = 'datasets/smalltalk.txt'
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
        await pro_memory.build_index()
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

        self._watcher_task = asyncio.create_task(_watch_datasets())
        self._running_tasks.append(self._watcher_task)

        async def _compression_worker() -> None:
            try:
                while True:
                    count = await pro_memory.total_adapter_usage()
                    if count >= COMPRESSION_INTERVAL:
                        try:
                            mut = LayerMutator(use_lora=True)
                            mut.load('adapter_pool')
                            lora_utils.prune_and_merge(mut.lora_layers)
                        except Exception as exc:  # pragma: no cover - logging side effect
                            logging.error("Compression failed: %s", exc)
                        await pro_memory.reset_adapter_usage()
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                raise

        self._compression_task = asyncio.create_task(_compression_worker())
        self._running_tasks.append(self._compression_task)
        self._start_dream_worker()

    def _start_dream_worker(self) -> None:
        if self._dream_task is None or self._dream_task.done():
            self._dream_task = asyncio.create_task(self._dream_worker())
            self._running_tasks.append(self._dream_task)

    async def _system_idle(self) -> bool:
        try:
            load = os.getloadavg()[0] / max(1, os.cpu_count() or 1)
        except OSError:
            load = 0.0
        if load > 0.2:
            return False
        if shutil.which("nvidia-smi") is None:
            return True
        try:
            out = await asyncio.to_thread(
                subprocess.check_output,
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
            )
            util = int(out.decode().splitlines()[0])
            if util > 10:
                return False
        except Exception:
            pass
        return True

    async def _dream_worker(self) -> None:
        try:
            while True:
                try:
                    await asyncio.wait_for(
                        self._dream_event.wait(), timeout=self._dream_interval
                    )
                    self._dream_event.clear()
                except asyncio.TimeoutError:
                    pass
                idle = await self._system_idle()
                if idle:
                    try:
                        await dream_mode.run(self)
                    except Exception:
                        pass
        except asyncio.CancelledError:
            pass

    def _start_tune_worker(self) -> None:
        if self._tune_worker_task is None or self._tune_worker_task.done():
            self.dataset_queue = asyncio.Queue()
            self._tune_worker_task = asyncio.create_task(self._tune_worker())
            self._running_tasks.append(self._tune_worker_task)

    async def _tune_worker(self) -> None:
        try:
            while True:
                path = await self.dataset_queue.get()
                try:
                    if path is None:
                        await self._async_tune([])
                    else:
                        await self._async_tune([path])
                finally:
                    self.dataset_queue.task_done()
        except asyncio.CancelledError:  # pragma: no cover - worker shutdown
            raise

    async def save_state(self) -> None:
        await asyncio.to_thread(pro_tune.save_state, self.state, STATE_PATH)

    async def scan_datasets(self) -> None:
        self._start_tune_worker()
        if not os.path.exists('datasets'):
            return
        old_hashes: Dict[str, str] = {}
        if os.path.exists(HASH_PATH):
            with open(HASH_PATH, 'r', encoding='utf-8') as fh:
                old_hashes = json.load(fh)
        new_hashes: Dict[str, str] = {}
        changed_files: List[str] = []
        weights_path = 'dataset_weights.json'
        if os.path.exists(weights_path):
            with open(weights_path, 'rb') as fh:
                new_hashes['__weights__'] = hashlib.sha256(fh.read()).hexdigest()
        dataset_names: List[str] = []
        paths: List[Tuple[str, str]] = []
        for name in os.listdir('datasets'):
            if name.endswith('.pkl'):
                continue
            path = os.path.join('datasets', name)
            if not os.path.isfile(path):
                continue
            dataset_names.append(name)
            paths.append((name, path))

        hash_semaphore = asyncio.Semaphore(SCAN_CONCURRENCY)

        def compute_hash(p: str) -> str:
            with open(p, 'rb') as fh:
                return hashlib.sha256(fh.read()).hexdigest()

        async def hash_file(name: str, path: str) -> Tuple[str, str, str]:
            async with hash_semaphore:
                digest = await asyncio.to_thread(compute_hash, path)
                return name, digest, path

        tasks = [hash_file(n, p) for n, p in paths]
        results = await asyncio.gather(*tasks)
        for name, digest, path in results:
            new_hashes[name] = digest
            if old_hashes.get(name) != digest:
                changed_files.append(path)
        removed = set(old_hashes) - set(new_hashes)
        weight_changed = old_hashes.get('__weights__') != new_hashes.get('__weights__')
        if removed and not new_hashes:
            with open(HASH_PATH, 'w', encoding='utf-8') as fh:
                json.dump(new_hashes, fh)
            await self.dataset_queue.put(None)
            return
        if removed or weight_changed:
            changed_files = [os.path.join('datasets', n) for n in dataset_names]
        with open(HASH_PATH, 'w', encoding='utf-8') as fh:
            json.dump(new_hashes, fh)
        for path in changed_files:
            await self.dataset_queue.put(path)

    async def _async_tune(self, paths: List[str]) -> None:
        tuned: List[str] = []
        weights: Dict[str, float] = {}
        if os.path.exists('dataset_weights.json'):
            try:
                with open('dataset_weights.json', 'r', encoding='utf-8') as fh:
                    weights = json.load(fh)
            except Exception as exc:  # pragma: no cover - logging side effect
                logging.error("Loading dataset weights failed: %s", exc)

        async def tune_path(path: str) -> None:
            async with self._tune_semaphore:
                try:
                    weight = float(weights.get(os.path.basename(path), 1.0))
                    adapters = self.select_adapters(path)
                    adapter_names = [n for n, _ in adapters]
                    await asyncio.to_thread(
                        pro_tune.train_weighted,
                        self.state,
                        path,
                        weight,
                        adapter_names,
                    )
                    tuned.append(os.path.basename(path))
                except Exception as exc:  # pragma: no cover - logging side effect
                    logging.error("Tuning failed for %s: %s", path, exc)

        await asyncio.gather(*(tune_path(p) for p in paths))
        try:
            await self.save_state()
        except Exception as exc:  # pragma: no cover - logging side effect
            logging.error("Saving state failed after tuning: %s", exc)
        if tuned:
            logging.info("Tuned datasets: %s", ", ".join(tuned))

    async def _maybe_spawn_specialist(self, dataset_path: str) -> None:
        """Spawn and merge a specialist if topic novelty is high."""
        if not self.last_forecast:
            return
        novelty = self.last_forecast.get("novelty", 0.0)
        if novelty <= self.novelty_threshold:
            return
        try:
            specialist = await asyncio.to_thread(
                pro_spawn.create_specialist, self.state, dataset_path
            )
            if specialist is not None:
                self.state = await asyncio.to_thread(
                    pro_tune.merge_specialist, self.state, specialist
                )
        except Exception as exc:  # pragma: no cover - logging side effect
            logging.error("Spawning specialist failed: %s", exc)

    async def shutdown(self) -> None:
        for task in self._running_tasks:
            task.cancel()
        if self._running_tasks:
            await asyncio.gather(*self._running_tasks, return_exceptions=True)
        await pro_predict.wait_save_task()
        await pro_meta.wait_recompute()

    def compute_charged_words(self, words: List[str]) -> List[str]:
        word_counts = Counter(words)
        charges: Dict[str, float] = {}
        for w, freq in word_counts.items():
            successors = len(self.state['bigram_counts'].get(w, {}))
            charges[w] = freq * (1 + successors)
        ordered = sorted(charges, key=charges.get, reverse=True)
        return ordered[:5]

    async def _forecast(self, seeds: List[str], depth: int = 2) -> None:
        """Generate a forecast tree of possible responses.

        The simulation explores continuations using :class:`MiniSelfAttention`
        and records the branch where entropy and resonance are closest.
        The result is stored in ``self.last_forecast``.
        """

        def _collect(node: pro_forecast.ForecastNode) -> List[pro_forecast.ForecastNode]:
            if not node.children:
                return [node]
            leaves: List[pro_forecast.ForecastNode] = []
            for child in node.children:
                leaves.extend(_collect(child))
            return leaves

        tree = await asyncio.to_thread(pro_forecast.simulate_paths, seeds, depth)
        leaves = _collect(tree)
        best = None
        best_score = -float("inf")
        for leaf in leaves:
            tokens = leaf.text.split()
            metrics = compute_metrics(
                tokens,
                self.state.get("trigram_counts", {}),
                self.state.get("bigram_counts", {}),
                self.state.get("word_counts", {}),
                self.state.get("char_ngram_counts", {}),
            )
            score = -abs(metrics.get("entropy", 0.0) - metrics.get("resonance", 0.0))
            if score > best_score:
                best = {"text": leaf.text, "prob": leaf.prob, "novelty": leaf.novelty}
                best_score = score
        self.last_forecast = best

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
        Additionally, no more than two single-letter words may appear
        consecutively and a sentence cannot end with two single letters.
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
                    if seq:
                        prev_lw = seq[-1].lower()
                        if (
                            (prev_lw == "a" and lw in {"you", "i"})
                            or (lw == "a" and prev_lw in {"you", "i"})
                        ):
                            continue
                        if len(cand) == 1 and len(seq[-1]) == 1:
                            if len(seq) >= 2 and len(seq[-2]) == 1:
                                continue
                            if len(seq) + 1 == target_length:
                                continue
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

    @timed
    async def respond(
        self,
        seeds: List[str],
        vocab: Optional[Dict[str, int]] = None,
        chaos_factor: Optional[float] = None,
        similarity_threshold: Optional[float] = None,
        forbidden: Optional[Set[str]] = None,
        update_meta: bool = True,
    ) -> str:
        start_time = time.perf_counter()
        if not seeds:
            duration = time.perf_counter() - start_time
            logging.info("respond took %.2fs", duration)
            return "Silence echoes within void."

        best = pro_meta.best_params()
        self.chaos_factor = best.get("chaos_factor", self.chaos_factor)
        self.similarity_threshold = best.get(
            "similarity_threshold", self.similarity_threshold
        )

        vocab = vocab or {}
        cf = self.chaos_factor if chaos_factor is None else chaos_factor
        similarity_threshold = (
            self.similarity_threshold
            if similarity_threshold is None
            else similarity_threshold
        )
        forbidden = {w.lower() for w in (forbidden or set())}
        analog_map: Dict[str, str] = {}
        analog_lock = asyncio.Lock()

        async def _build_analog(tok: str) -> None:
            suggestions = await pro_predict.suggest_async(tok, topn=1)
            analog = suggestions[0] if suggestions else None
            if not analog:
                analog = await asyncio.to_thread(pro_predict.lookup_analogs, tok)
            if analog:
                async with analog_lock:
                    analog_map[tok] = analog

        await asyncio.gather(*(_build_analog(tok) for tok in forbidden))
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
            def _metrics_and_length() -> Tuple[Dict, int]:
                tokens = [w.lower() for w in attempt_seeds if w]
                metrics_local = compute_metrics(
                    tokens,
                    self.state.get("trigram_counts", {}),
                    self.state.get("bigram_counts", {}),
                    self.state.get("word_counts", {}),
                    self.state.get("char_ngram_counts", {}),
                )
                return metrics_local, target_length_from_metrics(metrics_local)

            async def _lookup_seeds() -> List[Tuple[str, str]]:
                async def _lookup_seed(w: str) -> Tuple[str, str]:
                    analog = await asyncio.to_thread(
                        pro_predict.lookup_analogs, w.lower()
                    ) or w
                    if w.isupper():
                        analog = analog.upper()
                    elif w and w[0].isupper():
                        analog = analog[0].upper() + analog[1:]
                    return w, analog

                return await asyncio.gather(*(_lookup_seed(w) for w in attempt_seeds))

            async with asyncio.TaskGroup() as tg:
                metrics_task = tg.create_task(asyncio.to_thread(_metrics_and_length))
                seed_task = tg.create_task(_lookup_seeds())
                tg.create_task(self._forecast(attempt_seeds))

            metrics, target_length = metrics_task.result()
            seed_results = seed_task.result()
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

            words: List[str] = []
            tracker: Set[str] = set()
            for _, analog in seed_results:
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

            async def _lookup_inv(w: str) -> Tuple[str, Optional[str]]:
                analog = await asyncio.to_thread(pro_predict.lookup_analogs, w)
                return w, analog

            inv_results = await asyncio.gather(*(_lookup_inv(w) for w in high_inv_words))
            for w, analog in inv_results:
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
            words = await asyncio.to_thread(
                self.plan_sentence,
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
            await pro_predict._ensure_vectors()
            vectors = pro_predict._VECTORS

            def _compute_scores(
                first_words: List[str], tracker: Set[str], vectors: Dict[str, Dict[str, float]]
            ) -> Dict[str, float]:
                def _cos(a: Dict[str, float], b: Dict[str, float]) -> float:
                    keys = set(a) | set(b)
                    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
                    na = math.sqrt(sum(v * v for v in a.values()))
                    nb = math.sqrt(sum(v * v for v in b.values()))
                    if na == 0 or nb == 0:
                        return 0.0
                    return dot / (na * nb)

                first_vecs = [vectors[w] for w in first_words if w in vectors]
                scores_local: Dict[str, float] = {}
                for word, vec in vectors.items():
                    if word in tracker:
                        continue
                    if first_vecs:
                        sim = max(_cos(vec, fv) for fv in first_vecs)
                    else:
                        sim = 0.0
                    scores_local[word] = sim
                return scores_local

            scores = await asyncio.to_thread(
                _compute_scores, first_words, tracker, vectors
            )
            sim_thresh = similarity_threshold
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

            metrics_first = await asyncio.to_thread(
                compute_metrics,
                first_words,
                self.state.get("trigram_counts", {}),
                self.state.get("bigram_counts", {}),
                self.state.get("word_counts", {}),
                self.state.get("char_ngram_counts", {}),
            )
            target_length2 = await asyncio.to_thread(
                target_length_from_metrics,
                {
                    "entropy": metrics_first["entropy"],
                    "perplexity": metrics_first["perplexity"],
                },
                min_len=5,
                max_len=6,
            )
            words2 = await asyncio.to_thread(
                self.plan_sentence,
                second_seeds,
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
            if last1 in FORBIDDEN_ENDINGS:
                replacement = next(
                    (
                        w
                        for w in ordered
                        if w.lower() not in FORBIDDEN_ENDINGS
                    ),
                    last1,
                )
                parts = sentence1.rstrip(".").split()
                parts[-1] = replacement
                sentence1 = " ".join(parts) + "."

            last2 = sentence2.rstrip(".").split()[-1].lower()
            if last2 in FORBIDDEN_ENDINGS:
                replacement2 = next(
                    (
                        w
                        for w in ordered2
                        if w.lower() not in FORBIDDEN_ENDINGS
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
            if (
                grammar_filters.passes_filters(response)
                and await pro_memory.is_unique(response)
            ):
                await pro_memory.store_response(response)
                if update_meta:
                    resp_metrics = await asyncio.to_thread(
                        compute_metrics,
                        lowercase(tokenize(response)),
                        self.state.get("trigram_counts", {}),
                        self.state.get("bigram_counts", {}),
                        self.state.get("word_counts", {}),
                        self.state.get("char_ngram_counts", {}),
                    )
                    pro_meta.update(
                        resp_metrics,
                        {
                            "chaos_factor": cf,
                            "similarity_threshold": similarity_threshold,
                        },
                    )
                duration = time.perf_counter() - start_time
                level = logging.warning if duration > 5.0 else logging.info
                level("respond took %.2fs", duration)
                return response
            if extra_idx < len(ordered_vocab):
                attempt_seeds = list(attempt_seeds) + [ordered_vocab[extra_idx]]
            else:
                attempt_seeds = list(seeds) + [f"alt{extra_idx}"]
            extra_idx += 1

    async def prepare_candidates(self) -> None:
        """Generate candidate responses for recent messages."""
        try:
            recent = await pro_memory.fetch_recent_messages(5)
            new_cands = []
            for msg, emb in recent:
                seeds = tokenize(msg)
                for _ in range(2):
                    resp = await self.respond(seeds, update_meta=False)
                    new_cands.append((emb, resp))
            for cand in new_cands:
                self.candidate_buffer.append(cand)
        except Exception as exc:  # pragma: no cover - logging side effect
            logging.error("Preparing candidates failed: %s", exc)

    @timed
    async def process_message(self, message: str) -> Tuple[str, Dict]:
        best = pro_meta.best_params()
        self.chaos_factor = best.get("chaos_factor", self.chaos_factor)
        self.similarity_threshold = best.get(
            "similarity_threshold", self.similarity_threshold
        )
        original_words = tokenize(message)
        words = lowercase(original_words)
        adapter_pairs = self.select_adapters(message)
        adapters = [w for _, w in adapter_pairs]
        words = swap_pronouns(words)
        user_forbidden = set(words)
        async def _time(name: str, coro):
            start = time.perf_counter()
            try:
                return await coro
            finally:
                logging.info("%s took %.2fs", name, time.perf_counter() - start)

        msg_emb_task = asyncio.create_task(
            _time("embed_sentence", pro_rag_embedding.embed_sentence(message))
        )
        unknown: List[str] = [
            w for w in words if w not in self.state['word_counts']
        ]
        suggest_tasks = [pro_predict.suggest_async(w) for w in unknown]
        predicted: List[str] = []
        if suggest_tasks:
            for suggestions in await asyncio.gather(*suggest_tasks):
                predicted.extend(suggestions)
        # Blend n-gram prediction with transformer logits
        ngram_pred = ""
        if words:
            prev2 = words[-2] if len(words) >= 2 else ""
            prev1 = words[-1]
            ngram_pred = self.predict_next_word(prev2, prev1)
        trans_pred = ""
        vocab = list(self.state.get("word_counts", {}).keys())
        if vocab:
            att_tokens = self._drop_low_saliency(words[-5:])
            logits = await asyncio.to_thread(
                pro_predict.transformer_logits, att_tokens, vocab, adapters
            )
            trans_pred = max(logits, key=logits.get)
        blend: List[str] = []
        if ngram_pred:
            blend.append(ngram_pred)
        if trans_pred and trans_pred != ngram_pred:
            blend.append(trans_pred)
        predicted.extend(blend)
        mem_fetch = _time(
            "memory_fetch", pro_memory.fetch_similar_messages(message, top_k=5)
        )
        mem_encode = _time("memory_encode", pro_memory.encode_message(message))
        rag_retrieve = _time("rag_retrieve", pro_rag.retrieve(words))
        msg_emb, memory_context, mem_emb, context = await asyncio.gather(
            msg_emb_task, mem_fetch, mem_encode, rag_retrieve, return_exceptions=True
        )
        if isinstance(memory_context, Exception):
            logging.error("Memory retrieval failed: %s", memory_context)
            memory_context = []
        if isinstance(mem_emb, Exception):
            logging.error("Encoding message failed: %s", mem_emb)
            mem_emb = None
        if isinstance(context, Exception):
            logging.error("Context retrieval failed: %s", context)
            context = []
        ext_hits: List[str] = []
        if mem_emb is not None:
            try:
                ext_hits = await _time(
                    "vector_query", vector_store.query(mem_emb.tolist(), top_k=5)
                )
            except Exception as exc:  # pragma: no cover - logging side effect
                logging.error("External store query failed: %s", exc)
        memory_context.extend(ext_hits)
        store_tasks = []
        store_tasks.append(_time("memory_add", pro_memory.add_message(message)))
        if mem_emb is not None:
            store_tasks.append(
                _time(
                    "vector_upsert", vector_store.upsert(message, mem_emb.tolist())
                )
            )
        store_results = await asyncio.gather(*store_tasks, return_exceptions=True)
        for res in store_results:
            if isinstance(res, Exception):
                logging.error("Storing message failed: %s", res)
        context = memory_context + context
        try:
            if re.search(r"\b(AND|OR|NOT)\b", message):
                result = self.reasoner.evaluate(message, {w: True for w in words})
                predicted.append(str(result).lower())
        except Exception:
            pass
        context_tokens = tokenize(' '.join(context))
        all_words = words + lowercase(context_tokens)
        metrics = await asyncio.to_thread(
            compute_metrics,
            all_words,
            self.state['trigram_counts'],
            self.state['bigram_counts'],
            self.state['word_counts'],
            self.state['char_ngram_counts'],
        )
        seed_words = original_words + context_tokens + predicted
        recent_msgs, recent_resps = await pro_memory.fetch_recent(50)

        def _gather_tokens(texts: List[str]) -> List[str]:
            tokens: List[str] = []
            for text in texts:
                tokens.extend(lowercase(tokenize(text)))
            return tokens

        def _load_data_tokens() -> List[str]:
            tokens: List[str] = []
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
                                tokens.extend(lowercase(tokenize(line)))
                    except Exception:  # pragma: no cover - safety
                        continue
            return tokens

        mem_tokens_task = asyncio.to_thread(
            _gather_tokens, recent_msgs + recent_resps
        )
        data_tokens_task = asyncio.to_thread(_load_data_tokens)
        mem_tokens, data_tokens = await asyncio.gather(
            mem_tokens_task, data_tokens_task
        )

        def _combine_vocab(
            mem_tokens: List[str], data_tokens: List[str]
        ) -> Dict[str, int]:
            mem_counts = Counter(mem_tokens)
            data_counts = Counter(data_tokens)
            combined: Dict[str, int] = {}
            for w in set(mem_counts) | set(data_counts):
                weight = mem_counts.get(w, 0) + data_counts.get(w, 0)
                if w in mem_counts and w in data_counts:
                    weight *= 2
                combined[w] = weight
            return combined

        combined_vocab = await asyncio.to_thread(
            _combine_vocab, mem_tokens, data_tokens
        )
        best_candidate = None
        best_sim = 0.0
        for emb, resp in list(self.candidate_buffer):
            sim = float(np.dot(msg_emb, emb))
            if sim > best_sim:
                best_sim = sim
                best_candidate = (emb, resp)
        if best_candidate and best_sim > 0.9:
            response = best_candidate[1]
            try:
                self.candidate_buffer.remove(best_candidate)
            except ValueError:
                pass
        elif self.chaos_factor:
            response = await self.respond(
                seed_words,
                combined_vocab,
                chaos_factor=self.chaos_factor,
                forbidden=user_forbidden,
                update_meta=False,
            )
        else:
            response = await self.respond(
                seed_words,
                combined_vocab,
                forbidden=user_forbidden,
                update_meta=False,
            )
        try:
            await pro_memory.add_message(response)
            try:
                emb = await pro_memory.encode_message(response)
                await vector_store.upsert(response, emb.tolist())
            except Exception as exc:  # pragma: no cover - logging side effect
                logging.error("External store upsert failed: %s", exc)
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
            task = asyncio.create_task(self._async_tune([dataset_path]))
            self._tune_tasks.append(task)
            self._running_tasks.append(task)
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
        await pro_predict.enqueue_tokens(
            words + lowercase(tokenize(response))
        )
        vocab_list = list(self.state.get("word_counts", {}).keys())
        if vocab_list:
            asyncio.create_task(
                pro_predict.update_transformer(
                    vocab_list, recent_msgs, recent_resps
                )
            )
        try:
            await self.save_state()
        except Exception as exc:  # pragma: no cover - logging side effect
            logging.error("Saving state failed: %s", exc)
        asyncio.create_task(self.prepare_candidates())
        resp_metrics = await asyncio.to_thread(
            compute_metrics,
            lowercase(tokenize(response)),
            self.state['trigram_counts'],
            self.state['bigram_counts'],
            self.state['word_counts'],
            self.state['char_ngram_counts'],
        )
        pro_meta.update(
            resp_metrics,
            {
                "chaos_factor": self.chaos_factor,
                "similarity_threshold": self.similarity_threshold,
            },
        )
        await self.meta_controller.update(resp_metrics)
        self.log(message, response, metrics)
        await self._maybe_spawn_specialist(dataset_path)
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
        try:
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
        finally:
            await self.shutdown()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run the ProEngine chatbot")
    parser.add_argument(
        "--saliency-threshold",
        type=float,
        default=0.0,
        help="Percentile for dropping low-importance tokens before attention",
    )
    args = parser.parse_args()
    asyncio.run(
        ProEngine(saliency_threshold=args.saliency_threshold).interact()
    )
