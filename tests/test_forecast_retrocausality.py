import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import pro_forecast
import pro_predict


def test_forecast_retrocausality(monkeypatch):
    vocab = ["alpha", "beta", "gamma"]
    monkeypatch.setattr(pro_predict, "_VECTORS", {w: {} for w in vocab}, raising=False)
    monkeypatch.setattr(pro_predict, "_GRAPH", {w: {} for w in vocab}, raising=False)
    monkeypatch.setattr(pro_predict, "_TRANSFORMERS", {}, raising=False)
    monkeypatch.setattr(pro_predict, "_ensure_vectors", lambda: None, raising=False)

    root = pro_forecast.simulate_paths(["alpha"], depth=1)
    low_branch = min(root.children, key=lambda n: n.prob)
    target_word = low_branch.text.split()[-1]
    before = low_branch.prob

    pro_forecast.backpropagate_forecast(low_branch)

    root_after = pro_forecast.simulate_paths(["alpha"], depth=1)
    after = next(n.prob for n in root_after.children if n.text.endswith(target_word))
    assert after > before
