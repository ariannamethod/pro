import os
import sys
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pro_memory  # noqa: E402
import pro_rag  # noqa: E402
import pro_sequence  # noqa: E402
from pro_engine import ProEngine  # noqa: E402


def test_preserves_uppercase_mid_sentence(monkeypatch):
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

    response, _ = asyncio.run(engine.process_message("hello WORLD friend"))
    assert "WORLD" in response.split()[1:]
