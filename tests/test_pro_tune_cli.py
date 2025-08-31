import json
import subprocess


def test_pro_tune_cli(tmp_path):
    dataset = tmp_path / "data.txt"
    dataset.write_text("hello world")
    state_path = tmp_path / "state.json"
    subprocess.run([
        "python",
        "pro_tune.py",
        str(dataset),
        "--state-path",
        str(state_path),
    ], check=True)
    assert state_path.exists()
    data = json.loads(state_path.read_text())
    assert data["word_counts"].get("hello", 0) == 1


def test_fractal_links_converge():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from trainer import fractal_links
    from self_reflect.trainer import SelfFineTuner

    # Link two dummy layers and prepare a simple model with distinct weights
    previous_links = fractal_links.copy()
    fractal_links.clear()
    fractal_links.append(("w1", "w2"))

    class Dummy:
        def __init__(self) -> None:
            self.w1 = 0.0
            self.w2 = 10.0
            self.resonance = 0.5

    model = Dummy()
    tuner = SelfFineTuner(model)
    before = abs(model.w1 - model.w2)
    tuner.run(["??"], {})
    after = abs(model.w1 - model.w2)

    fractal_links[:] = previous_links
    assert after < before
