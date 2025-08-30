import asyncio
import json
import logging
import os
import threading
import time

import pro_engine
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


def test_scan_triggers_tune_on_weight_change(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    data_file = datasets_dir / "sample.txt"
    data_file.write_text("one")
    weights_file = tmp_path / "dataset_weights.json"
    weights_file.write_text(json.dumps({"sample.txt": 1}))

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
    weights_file.write_text(json.dumps({"sample.txt": 2}))
    asyncio.run(run_scan())
    assert called == [expected]


def test_async_tune_logs_and_saves(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_file = tmp_path / "sample.txt"
    data_file.write_text("data")
    weights_file = tmp_path / "dataset_weights.json"
    weights_file.write_text(json.dumps({"sample.txt": 2}))

    engine = ProEngine()

    trained = []

    def fake_train(state, path, weight):
        trained.append((path, weight))

    monkeypatch.setattr(pro_tune, "train_weighted", fake_train)

    saved = []

    async def fake_save_state():
        saved.append(True)

    monkeypatch.setattr(engine, "save_state", fake_save_state)

    logs = []

    def fake_info(msg, *args):
        logs.append(msg % args)

    monkeypatch.setattr(logging, "info", fake_info)

    asyncio.run(engine._async_tune([str(data_file)]))

    assert trained == [(str(data_file), 2)]
    assert saved
    assert any("sample.txt" in entry for entry in logs)


def test_async_tune_runs_concurrently(tmp_path, monkeypatch):
    monkeypatch.setattr(pro_engine, "TUNE_CONCURRENCY", 2)

    data_files = []
    for name in ["one.txt", "two.txt", "three.txt"]:
        path = tmp_path / name
        path.write_text("data")
        data_files.append(str(path))

    engine = ProEngine()

    running = 0
    max_running = 0
    lock = threading.Lock()

    def fake_train(state, path, weight):
        nonlocal running, max_running
        with lock:
            running += 1
            max_running = max(max_running, running)
        time.sleep(0.1)
        with lock:
            running -= 1

    monkeypatch.setattr(pro_tune, "train_weighted", fake_train)

    async def fake_save_state():
        pass

    monkeypatch.setattr(engine, "save_state", fake_save_state)

    asyncio.run(engine._async_tune(data_files))

    assert max_running > 1
    assert max_running <= 2
