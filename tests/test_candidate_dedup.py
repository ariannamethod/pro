import numpy as np
import types
import sys

# Stub external dependencies to allow importing pro_engine without heavy requirements
for name in [
    "pro_tune",
    "pro_sequence",
    "pro_memory",
    "pro_rag",
    "pro_rag_embedding",
    "pro_predict",
    "pro_forecast",
    "pro_meta",
    "dream_mode",
    "lora_utils",
    "grammar_filters",
    "message_utils",
    "pro_spawn",
]:
    sys.modules.setdefault(name, types.ModuleType(name))

# Minimal stubs for modules with attributes used by pro_engine
metrics = types.ModuleType("pro_metrics")
metrics.tokenize = lambda *a, **k: []
metrics.compute_metrics = lambda *a, **k: {}
metrics.lowercase = lambda x: x
metrics.target_length_from_metrics = lambda *a, **k: 0
sys.modules.setdefault("pro_metrics", metrics)

identity = types.ModuleType("pro_identity")
identity.swap_pronouns = lambda w: w
sys.modules.setdefault("pro_identity", identity)

autoadapt = types.ModuleType("autoadapt")
class LayerMutator:  # pragma: no cover - simple stub
    pass
autoadapt.LayerMutator = LayerMutator
sys.modules.setdefault("autoadapt", autoadapt)

watchfiles = types.ModuleType("watchfiles")
watchfiles.awatch = lambda *a, **k: None
sys.modules.setdefault("watchfiles", watchfiles)

transformers_blocks = types.ModuleType("transformers.blocks")
class SymbolicReasoner:  # pragma: no cover - simple stub
    pass
class LightweightMoEBlock:  # pragma: no cover - simple stub
    def __init__(self, *a, **k):
        pass
transformers_blocks.SymbolicReasoner = SymbolicReasoner
transformers_blocks.LightweightMoEBlock = LightweightMoEBlock
transformers_pkg = types.ModuleType("transformers")
transformers_pkg.blocks = transformers_blocks
sys.modules.setdefault("transformers", transformers_pkg)
sys.modules.setdefault("transformers.blocks", transformers_blocks)

meta = types.ModuleType("meta_controller")
class MetaController:  # pragma: no cover - simple stub
    def __init__(self, *a, **k):
        pass
meta.MetaController = MetaController
sys.modules.setdefault("meta_controller", meta)

api_mod = types.ModuleType("api")
api_mod.vector_store = types.ModuleType("vector_store")
sys.modules.setdefault("api", api_mod)

# Provide required attributes for specific stubs
sys.modules["pro_tune"].train = lambda *a, **k: None

import pro_engine


def test_filter_similar_candidates_removes_duplicates():
    emb1 = np.array([1.0, 0.0], dtype=float)
    emb2 = np.array([0.99, 0.01], dtype=float)
    emb3 = np.array([-1.0, 0.0], dtype=float)
    cands = [(emb1, "foo"), (emb2, "foo"), (emb3, "bar")]
    filtered = pro_engine.filter_similar_candidates(cands, threshold=0.98)
    texts = [t for _, t in filtered]
    assert len(filtered) == 2
    assert len(set(texts)) == len(texts)


def test_filter_similar_candidates_handles_many_duplicates():
    emb_foo = np.array([1.0, 0.0], dtype=float)
    emb_bar = np.array([-1.0, 0.0], dtype=float)
    cands = [
        (emb_foo, "foo"),
        (emb_foo, "foo"),
        (emb_foo, "foo"),
        (emb_bar, "bar"),
    ]
    filtered = pro_engine.filter_similar_candidates(cands, threshold=0.98)
    assert [t for _, t in filtered] == ["foo", "bar"]


def test_rank_candidates_no_duplicate_topn():
    engine = pro_engine.ProEngine()
    engine.candidate_buffer.clear()
    emb1 = np.array([1.0, 0.0], dtype=float)
    emb2 = np.array([0.99, 0.01], dtype=float)
    emb3 = np.array([-1.0, 0.0], dtype=float)
    engine.candidate_buffer.append((emb1, "foo"))
    engine.candidate_buffer.append((emb2, "foo"))
    engine.candidate_buffer.append((emb3, "bar"))
    ranked = engine.rank_candidates(np.array([1.0, 0.0], dtype=float), topn=2)
    responses = [resp for _, _, resp in ranked]
    assert len(responses) == 2
    assert len(set(responses)) == len(responses)


def test_rank_candidates_cosine_dedup_after_sort():
    engine = pro_engine.ProEngine()
    engine.candidate_buffer.clear()
    emb1 = np.array([0.5, 0.0], dtype=float)
    emb2 = np.array([0.499, 0.0], dtype=float)
    emb3 = np.array([-0.5, 0.0], dtype=float)
    engine.candidate_buffer.append((emb1, "foo"))
    engine.candidate_buffer.append((emb2, "bar"))
    engine.candidate_buffer.append((emb3, "baz"))
    ranked = engine.rank_candidates(np.array([1.0, 0.0], dtype=float), topn=2)
    responses = [resp for _, _, resp in ranked]
    assert responses == ["foo", "baz"]
