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
