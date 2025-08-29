import asyncio
import logging
import os
from pro_engine import ProEngine
import pro_tune


def test_scan_triggers_tune_on_change(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    data_file = datasets_dir / "sample.txt"
    data_file.write_text("one")

    engine = ProEngine()
    called = []

    async def fake_async_tune(paths):
        called.extend(paths)

    monkeypatch.setattr(engine, "_async_tune", fake_async_tune)

    async def run_scan():
        await engine.scan_datasets()
        await asyncio.sleep(0)

    asyncio.run(run_scan())
    expected = os.path.join("datasets", "sample.txt")
    assert called == [expected]

    called.clear()
    data_file.write_text("two")
    asyncio.run(run_scan())
    assert called == [expected]


def test_async_tune_logs_and_saves(tmp_path, monkeypatch):
    data_file = tmp_path / "sample.txt"
    data_file.write_text("data")

    engine = ProEngine()

    trained = []

    def fake_train(state, path):
        trained.append(path)

    monkeypatch.setattr(pro_tune, "train", fake_train)

    saved = []

    async def fake_save_state():
        saved.append(True)

    monkeypatch.setattr(engine, "save_state", fake_save_state)

    logs = []

    def fake_info(msg, *args):
        logs.append(msg % args)

    monkeypatch.setattr(logging, "info", fake_info)

    asyncio.run(engine._async_tune([str(data_file)]))

    assert trained == [str(data_file)]
    assert saved
    assert any("sample.txt" in entry for entry in logs)
