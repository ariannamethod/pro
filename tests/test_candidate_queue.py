import asyncio
import numpy as np

import pro_engine
import pro_memory
import pro_predict
import pro_rag
import pro_rag_embedding
import pro_sequence
import pro_meta
import builtins


class DummyVectorStore:
    async def query(self, *args, **kwargs):
        return []

    async def upsert(self, *args, **kwargs):
        return None


def _no_io_open(*args, **kwargs):
    mode = kwargs.get("mode", "r")
    class Dummy:
        def write(self, *a, **k):
            pass
        def read(self, *a, **k):
            return b"" if "b" in mode else ""
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            pass
    return Dummy()
def _noop_sync(*args, **kwargs):
    return {"entropy": 0.0, "perplexity": 0.0}


def test_candidate_generation_sequential(monkeypatch):
    engine = pro_engine.ProEngine()

    # Patch heavy dependencies to no-op
    monkeypatch.setattr(pro_rag_embedding, "embed_sentence", lambda m: asyncio.sleep(0, result=np.zeros(1)))
    monkeypatch.setattr(pro_predict, "suggest_async", lambda w: asyncio.sleep(0, result=[]))
    monkeypatch.setattr(pro_predict, "transformer_logits", lambda t, v, a: {})
    monkeypatch.setattr(pro_memory, "fetch_similar_messages", lambda m, top_k=5: asyncio.sleep(0, result=[]))
    monkeypatch.setattr(pro_memory, "encode_message", lambda m: asyncio.sleep(0, result=np.zeros(1)))
    monkeypatch.setattr(pro_rag, "retrieve", lambda w: asyncio.sleep(0, result=[]))
    monkeypatch.setattr(pro_memory, "add_message", lambda m: asyncio.sleep(0))
    monkeypatch.setattr(pro_memory, "fetch_recent", lambda n: asyncio.sleep(0, result=([], [])))
    monkeypatch.setattr(pro_predict, "enqueue_tokens", lambda toks: asyncio.sleep(0))
    monkeypatch.setattr(pro_predict, "update_transformer", lambda v, m, r: asyncio.sleep(0))
    monkeypatch.setattr(pro_sequence, "analyze_sequences", lambda *a, **k: None)
    monkeypatch.setattr(pro_engine, "compute_metrics", _noop_sync)
    monkeypatch.setattr(engine, "save_state", lambda : asyncio.sleep(0))
    monkeypatch.setattr(engine, "respond", lambda *a, **k: asyncio.sleep(0, result="ok"))
    monkeypatch.setattr(engine, "_async_tune", lambda paths: asyncio.sleep(0))
    monkeypatch.setattr(pro_meta, "best_params", lambda: {})
    monkeypatch.setattr(pro_engine, "vector_store", DummyVectorStore())
    monkeypatch.setattr(builtins, "open", _no_io_open)
    monkeypatch.setattr(pro_engine.os.path, "exists", lambda p: False)
    monkeypatch.setattr(pro_engine.os, "makedirs", lambda *a, **k: None)
    monkeypatch.setattr(pro_engine.json, "load", lambda f: {})
    monkeypatch.setattr(pro_engine.json, "dump", lambda obj, f: None)

    running = 0
    overlaps = 0

    async def tracked_prepare(self):
        nonlocal running, overlaps
        if running:
            overlaps += 1
        running += 1
        await asyncio.sleep(0.01)
        running -= 1

    monkeypatch.setattr(engine, "prepare_candidates", tracked_prepare.__get__(engine, pro_engine.ProEngine))

    async def run_msgs():
        tasks = [asyncio.create_task(engine.process_message(f"msg {i}")) for i in range(10)]
        await asyncio.gather(*tasks)
        await engine._candidate_queue.join()

    asyncio.run(run_msgs())
    assert overlaps == 0
