import asyncio
import pytest
import pro_engine
import pro_memory
import lora_utils
import pro_tune


@pytest.mark.asyncio
async def test_compression_runs_after_threshold(monkeypatch, tmp_path):
    pro_memory.DB_PATH = str(tmp_path / "mem.db")
    pro_memory.COMPRESSION_INTERVAL = 2

    called = 0

    def fake_prune(layers):  # pragma: no cover - simple stub
        nonlocal called
        called += 1
        return {}

    monkeypatch.setattr(lora_utils, "prune_and_merge", fake_prune)

    async def dummy_awatch(path):
        while True:
            await asyncio.sleep(3600)
            yield set()

    monkeypatch.setattr(pro_engine, "awatch", dummy_awatch)

    class DummyMut:
        def __init__(self, use_lora=True):
            self.lora_layers = {}

        def load(self, path):
            pass

    monkeypatch.setattr(pro_engine, "LayerMutator", DummyMut)
    monkeypatch.setattr(pro_tune, "train", lambda state, path: None)

    engine = pro_engine.ProEngine()
    await engine.setup()
    await pro_memory.reset_adapter_usage()

    await pro_memory.increment_adapter_usage("a1")
    await asyncio.sleep(0.05)
    assert called == 0

    await pro_memory.increment_adapter_usage("a2")
    await asyncio.sleep(0.05)
    assert called == 1

    await engine.shutdown()
    await pro_memory.close_db()
