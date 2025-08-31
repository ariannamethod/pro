import os
import sys
import asyncio
from pathlib import Path

import pytest

# Ensure token for module import
os.environ.setdefault("TELEGRAM_TOKEN", "TOKEN")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pro_tg  # noqa: E402
import pro_engine  # noqa: E402


class DummyEngine:
    async def setup(self) -> None:  # pragma: no cover - simple stub
        pass

    async def shutdown(self) -> None:  # pragma: no cover - simple stub
        pass

    async def process_message(self, text: str):  # pragma: no cover - simple stub
        return "", None


@pytest.mark.asyncio
async def test_exponential_backoff(monkeypatch):
    monkeypatch.setattr(pro_engine, "ProEngine", DummyEngine)

    call_state = {"count": 0}

    async def fake_get_updates(session, offset):  # noqa: ARG001
        call_state["count"] += 1
        if call_state["count"] in (1, 2):
            raise RuntimeError("boom")
        if call_state["count"] == 3:
            return []
        if call_state["count"] == 4:
            raise RuntimeError("boom")
        raise SystemExit

    monkeypatch.setattr(pro_tg, "get_updates", fake_get_updates)

    delays: list[float] = []

    async def fake_sleep(delay: float):  # noqa: ARG001
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    with pytest.raises(SystemExit):
        await pro_tg.main()

    assert delays == [1, 2, 1]
