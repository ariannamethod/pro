import os
import sys
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
async def test_immediate_retry_limit(monkeypatch):
    monkeypatch.setattr(pro_engine, "ProEngine", DummyEngine)

    call_state = {"count": 0}

    async def fake_get_updates(session, offset):  # noqa: ARG001
        call_state["count"] += 1
        raise RuntimeError("boom")

    monkeypatch.setattr(pro_tg, "get_updates", fake_get_updates)
    monkeypatch.setattr(pro_tg, "MAX_RETRIES", 3)

    with pytest.raises(RuntimeError):
        await pro_tg.main()

    assert call_state["count"] == 3
