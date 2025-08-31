import os
import sys
import asyncio

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pro_memory  # noqa: E402
import pro_rag  # noqa: E402
import pro_sequence  # noqa: E402
import pro_predict  # noqa: E402
from pro_engine import ProEngine  # noqa: E402


@pytest.fixture
def engine(monkeypatch):
    engine = ProEngine()

    async def dummy_add_message(*args, **kwargs):
        pass

    async def dummy_retrieve(*args, **kwargs):
        return []

    async def dummy_save_state(*args, **kwargs):
        pass

    monkeypatch.setattr(pro_memory, "add_message", dummy_add_message)
    monkeypatch.setattr(pro_rag, "retrieve", dummy_retrieve)
    monkeypatch.setattr(engine, "save_state", dummy_save_state)

    def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(pro_sequence, "analyze_sequences", noop)

    mapping = {
        "hello": ["hi"],
        "world": ["globe"],
        "friend": ["buddy"],
        "nasa": ["space"],
        "launches": ["starts"],
        "rockets": ["missiles"],
        "iphone": ["smartphone"],
        "release": ["launch"],
        "soon": ["shortly"],
    }

    async def fake_suggest_async(w, topn=3, _m=mapping):
        return _m.get(w.lower(), [])

    monkeypatch.setattr(pro_predict, "suggest_async", fake_suggest_async)
    monkeypatch.setattr(
        pro_predict, "suggest", lambda w, topn=3, _m=mapping: _m.get(w.lower(), [])
    )
    monkeypatch.setattr(pro_predict, "lookup_analogs", lambda w: None)
    monkeypatch.setattr(pro_predict, "_VECTORS", {"foo": {"a": 1.0}})

    return engine


def test_preserves_uppercase_mid_sentence(engine):
    response, _ = asyncio.run(engine.process_message("hello WORLD friend"))
    assert "GLOBE" in response.split()[1:]
    assert "WORLD" not in response


def test_acronym_first_word(engine):
    response, _ = asyncio.run(engine.process_message("NASA launches rockets"))
    assert response.split()[0] == "SPACE"
    assert "NASA" not in response


def test_mixed_case_first_word(engine):
    response, _ = asyncio.run(engine.process_message("iPhone release soon"))
    assert response.split()[0] == "Smartphone"
    assert "iPhone" not in response
