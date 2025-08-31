from pathlib import Path
import shutil

from trainer import Trainer


def test_meta_optimizer_updates_params():
    log_dir = Path("logs/meta")
    if log_dir.exists():
        shutil.rmtree(log_dir)
    trainer = Trainer(eval_interval=1)
    trainer.params = {"w": 1.0}
    conversations = ["??"]
    trainer.train_step("w", 1.0, conversations=conversations)
    assert trainer.params["w"] > 1.0
    before_file = log_dir / "epoch_1_before.json"
    after_file = log_dir / "epoch_1_after.json"
    assert before_file.exists()
    assert after_file.exists()
