from pro_engine import ProEngine
from transformers import blocks


def test_self_growing_block_loading():
    engine = ProEngine()

    assert hasattr(blocks, "SpawnBlock")
    block = blocks.SpawnBlock()
    assert block.forward(3) == 6
