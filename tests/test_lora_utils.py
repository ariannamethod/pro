import pytest
from autoadapt import LoRALayer
import lora_utils


def test_prune_and_merge():
    layer = LoRALayer(
        name="dense",
        rank=2,
        alpha=1.0,
        matrix_a=[[1.0, 0.0], [0.0, 1.0]],
        matrix_b=[[0.5, 0.0], [0.0, 0.0001]],
    )
    merged = lora_utils.prune_and_merge({"dense": layer}, threshold=0.001)
    assert merged["dense"][0][0] == pytest.approx(0.25)
    assert merged["dense"][1][1] == 0.0
