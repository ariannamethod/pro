from trainer import Trainer


def test_mini_model_distillation() -> None:
    trainer = Trainer()
    dialogue = ["hello", "there", "general", "kenobi"]
    trainer.evolve("layer_x", dialogue, metric=0.9)
    assert "layer_x" in trainer.mutator.mutations
