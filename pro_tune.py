import logging
import json
import argparse
import os
import asyncio
from typing import Dict

from pro_metrics import tokenize, lowercase
import pro_sequence
import pro_predict
import pro_memory
from pro_rag import retrieve_external

STATE_PATH = 'pro_state.json'
_SEP = '\u0001'


def train_weighted(
    state: Dict, dataset_path: str, weight: float, adapters: list[str] | None = None
) -> Dict:
    if weight <= 0:
        logging.warning(
            "Non-positive weight %s for %s; skipping", weight, dataset_path
        )
        return state
    if not os.path.exists(dataset_path):
        logging.warning(
            "Dataset path %s does not exist; skipping training", dataset_path
        )
        return state
    with open(dataset_path, 'r', encoding='utf-8') as fh:
        text = fh.read()
    if not text:
        logging.warning(
            "Dataset path %s is empty; skipping training", dataset_path
        )
        return state
    words = lowercase(tokenize(text))
    pro_sequence.analyze_sequences(state, words, weight=weight)
    asyncio.run(pro_predict.update(words))
    if adapters:
        for name in adapters:
            try:
                asyncio.run(pro_memory.increment_adapter_usage(name))
            except Exception:
                pass
    pro_predict.save_embeddings(pro_predict._GRAPH, pro_predict._VECTORS)
    return state


def train(
    state: Dict, dataset_path: str, adapters: list[str] | None = None
) -> Dict:
    return train_weighted(state, dataset_path, 1.0, adapters)


async def tune_with_knowledge(
    state: Dict, query: str, source: str = "wikipedia", weight: float = 1.0
) -> Dict:
    """Retrieve external knowledge by *query* and fine-tune *state* on it."""
    docs = await retrieve_external(query, source)
    if not docs:
        return state
    text = " ".join(docs)
    words = lowercase(tokenize(text))
    pro_sequence.analyze_sequences(state, words, weight=weight)
    await pro_predict.update(words)
    await asyncio.to_thread(
        pro_predict.save_embeddings, pro_predict._GRAPH, pro_predict._VECTORS
    )
    return state


def merge_specialist(
    base_state: Dict, specialist_state: Dict, temperature: float = 0.5
) -> Dict:
    """Merge ``specialist_state`` into ``base_state`` using a weighted blend.

    Parameters
    ----------
    base_state:
        Original model state to be updated in-place.
    specialist_state:
        Fine-tuned specialist whose knowledge should be distilled.
    temperature:
        Weighting factor giving preference to the specialist.  ``0.5`` blends
        both equally, while values closer to ``1.0`` favor the specialist.
    """

    if specialist_state is None:
        return base_state
    weight = float(temperature)
    for key in [
        "word_counts",
        "bigram_counts",
        "trigram_counts",
        "char_ngram_counts",
    ]:
        base = base_state.setdefault(key, {})
        spec = specialist_state.get(key, {})
        for k, v in spec.items():
            base[k] = base.get(k, 0.0) * (1.0 - weight) + v * weight
    return base_state


def _serialize_state(state: Dict) -> Dict:
    data = dict(state)
    for k in ['word_inv', 'bigram_inv', 'trigram_inv', 'char_ngram_inv']:
        data.pop(k, None)
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
    parser.add_argument(
        "--knowledge-query",
        help="Query string for external knowledge retrieval",
    )
    parser.add_argument(
        "--knowledge-source",
        default="wikipedia",
        help="External knowledge source (default: wikipedia)",
    )
    parser.add_argument(
        "--knowledge-weight",
        type=float,
        default=1.0,
        help="Training weight for retrieved knowledge",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    state = load_state(args.state_path)
    state = train(state, args.dataset_path)
    if args.knowledge_query:
        state = asyncio.run(
            tune_with_knowledge(
                state,
                args.knowledge_query,
                source=args.knowledge_source,
                weight=args.knowledge_weight,
            )
        )
    save_state(state, args.state_path)
    logging.info("Training complete for %s", args.dataset_path)
