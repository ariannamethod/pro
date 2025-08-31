import asyncio
import pytest

import pro_engine
from pro_engine import ProEngine


@pytest.mark.asyncio
async def test_system_idle_true(monkeypatch):
    samples = iter([(100, 200), (190, 300)])

    def fake_read_cpu_times():
        return next(samples)

    monkeypatch.setattr(pro_engine, "_read_cpu_times", fake_read_cpu_times)
    engine = ProEngine()
    assert await engine._system_idle()


@pytest.mark.asyncio
async def test_system_idle_false(monkeypatch):
    samples = iter([(100, 200), (110, 220)])

    def fake_read_cpu_times():
        return next(samples)

    monkeypatch.setattr(pro_engine, "_read_cpu_times", fake_read_cpu_times)
    engine = ProEngine()
    assert not await engine._system_idle()
