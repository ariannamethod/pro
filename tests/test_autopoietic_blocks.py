import autoadapt
from pro_engine import ProEngine
from transformers import blocks


def test_block_generation_and_loading():
    code = autoadapt.generate_block_code("MorphBlock", scale=3.0)
    assert "class MorphBlock" in code

    engine = ProEngine()
    engine.load_generated_block(code, "MorphBlock")

    assert hasattr(blocks, "MorphBlock")
    block = blocks.MorphBlock()
    assert block.forward(2) == 6
