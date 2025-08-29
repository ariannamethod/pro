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

    def fake_create_task(coro):
        calls.append(coro)
        coro.close()
        loop = asyncio.get_running_loop()
        return loop.create_task(asyncio.sleep(0))

    monkeypatch.setattr(asyncio, "create_task", fake_create_task)

    asyncio.run(engine.scan_datasets())
    calls.clear()

    data_file.unlink()

    asyncio.run(engine.scan_datasets())
    assert len(calls) == 1

