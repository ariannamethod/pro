import asyncio
import pytest

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pro_memory  # noqa: E402
import pro_predict  # noqa: E402
import message_utils  # noqa: E402


@pytest.mark.asyncio
async def test_build_analog_map(monkeypatch):
    mapping = {"hello": "hi", "world": "earth"}

    async def fake_suggest_async(w, topn=1, _m=mapping):
        return [_m[w]] if w in _m else []

    monkeypatch.setattr(pro_predict, "suggest_async", fake_suggest_async)
    monkeypatch.setattr(pro_predict, "lookup_analogs", lambda w: mapping.get(w))

    analog_map = await message_utils.build_analog_map(mapping.keys())
    assert analog_map == mapping

    word = "World"
    repl = analog_map[word.lower()]
    if word.isupper():
        repl = repl.upper()
    elif word and word[0].isupper():
        repl = repl[0].upper() + repl[1:]
    assert repl == "Earth"


@pytest.mark.asyncio
async def test_ensure_unique(tmp_path, monkeypatch):
    db_path = tmp_path / "mem.db"
    monkeypatch.setattr(pro_memory, "DB_PATH", str(db_path))
    await pro_memory.init_db()
    assert await message_utils.ensure_unique("hello world")
    assert not await message_utils.ensure_unique("hello world")
