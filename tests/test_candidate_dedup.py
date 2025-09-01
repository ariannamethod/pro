import numpy as np

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
