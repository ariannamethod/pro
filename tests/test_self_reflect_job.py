import asyncio
import contextlib
import importlib.util
import pathlib
import sys

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

SPEC = importlib.util.spec_from_file_location(
    "self_reflect_job",
    pathlib.Path(__file__).resolve().parents[1] / "cron" / "self_reflect_job.py",
)
job = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(job)


@pytest.mark.asyncio
async def test_schedule_multiple_cycles(monkeypatch):
    calls = 0
    done = asyncio.Event()

    async def fake_run_cycle(conv=None):
        nonlocal calls
        calls += 1
        if calls >= 2:
            done.set()

    monkeypatch.setattr(job, "run_cycle", fake_run_cycle)
    job.INTERVAL_SECONDS = 0

    loop = asyncio.get_running_loop()
    handle = job.schedule_next(loop=loop)

    await asyncio.wait_for(done.wait(), timeout=0.1)

    handle.cancel()
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for t in pending:
        t.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await t

    assert calls >= 2
