import os
import sys
import asyncio


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pro_engine import ProEngine  # noqa: E402


def test_retrain_triggered_on_dataset_file_removal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    data_file = datasets_dir / "sample.txt"
    data_file.write_text("data")

    engine = ProEngine()
    calls = []

    async def fake_async_tune(paths):
        calls.append(paths)

    monkeypatch.setattr(engine, "_async_tune", fake_async_tune)

    async def run_scan():
        await engine.scan_datasets()
        await asyncio.sleep(0)

    asyncio.run(run_scan())
    calls.clear()

    data_file.unlink()

    asyncio.run(run_scan())
    assert calls == [[]]
