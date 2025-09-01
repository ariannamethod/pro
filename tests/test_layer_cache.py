import json

from memory import cache_layers
from memory import layer_cache


def test_cache_layers_creates_and_reuses(tmp_path, monkeypatch):
    macros = {"dense": {"in": 2, "out": 3}}
    monkeypatch.setattr(layer_cache, "_CACHE_DIR", tmp_path)
    first = cache_layers(macros)
    second = cache_layers(macros)
    assert first == second
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    with open(files[0], "r", encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert on_disk == first
