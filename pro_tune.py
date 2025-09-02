import logging
import json
import argparse
import os
import asyncio
from typing import Dict, List, Optional

from pro_metrics import tokenize, lowercase
import pro_sequence
import pro_predict
import pro_memory
from pro_rag import retrieve_external

STATE_PATH = 'pro_state.json'
_SEP = '\u0001'


def train_weighted(
    state: Dict, dataset_path: str, weight: float, adapters: Optional[List[str]] = None,
    message_metrics: Optional[Dict] = None
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
    # Умная загрузка - случайный участок + семантический поиск
    file_size = os.path.getsize(dataset_path)
    texts = []
    
    # 1. Случайный участок на основе метрик
    with open(dataset_path, 'r', encoding='utf-8') as fh:
        if message_metrics:
            # Размер участка зависит от энтропии сообщения
            entropy = message_metrics.get('entropy', 0.5)
            perplexity = message_metrics.get('perplexity', 1.0)
            
            # Чем больше энтропия - тем больше участок (больше разнообразия)
            chunk_size = int(1000 + entropy * 2000)  # 1000-3000 символов
            
            # Позиция зависит от перплексии (заряженности)
            position_factor = (perplexity % 1.0)  # 0.0 - 1.0
            max_start = max(0, file_size - chunk_size)
            start_pos = int(max_start * position_factor)
        else:
            # Без метрик - случайный участок
            import random
            chunk_size = random.randint(800, 1200)
            max_start = max(0, file_size - chunk_size)
            start_pos = random.randint(0, max_start)
        
        fh.seek(start_pos)
        random_text = fh.read(chunk_size)
        if random_text.strip():
            texts.append(random_text)
    
    # 2. Семантический поиск (если есть метрики с исходными словами)
    if message_metrics and 'words' in message_metrics:
        query_words = message_metrics['words']
        # Синхронная версия семантического поиска
        semantic_chunks = _find_semantic_chunks_sync(dataset_path, query_words, num_chunks=1)
        texts.extend(semantic_chunks)
    
    # Объединяем все тексты
    text = " ".join(texts) if texts else ""
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
    state: Dict, dataset_path: str, adapters: Optional[List[str]] = None,
    message_metrics: Optional[Dict] = None
) -> Dict:
    return train_weighted(state, dataset_path, 1.0, adapters, message_metrics)


def _find_semantic_chunks_sync(
    dataset_path: str, query_words: List[str], num_chunks: int = 2, chunk_size: int = 1000
) -> List[str]:
    """Найти семантически релевантные куски датасета (синхронная версия)."""
    if not os.path.exists(dataset_path):
        return []
    
    with open(dataset_path, 'r', encoding='utf-8') as fh:
        full_text = fh.read()
    
    # Разбиваем на куски
    chunks = []
    for i in range(0, len(full_text), chunk_size // 2):  # с перекрытием
        chunk = full_text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    
    if not chunks:
        return []
    
    # Ищем наиболее релевантные куски
    query_set = set(w.lower() for w in query_words)
    scored_chunks = []
    
    for chunk in chunks:
        chunk_words = set(w.lower() for w in lowercase(tokenize(chunk)))
        # Семантическая близость = пересечение слов
        overlap = len(query_set & chunk_words)
        if overlap > 0:
            scored_chunks.append((overlap, chunk))
    
    # Возвращаем топ куски
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored_chunks[:num_chunks]]


async def find_semantic_chunks(
    dataset_path: str, query_words: List[str], num_chunks: int = 2, chunk_size: int = 1000
) -> List[str]:
    """Найти семантически релевантные куски датасета (асинхронная версия)."""
    from compat import to_thread
    return await to_thread(_find_semantic_chunks_sync, dataset_path, query_words, num_chunks, chunk_size)


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
    from compat import to_thread
    await to_thread(
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
