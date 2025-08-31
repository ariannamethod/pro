import pathlib
import sys
import numpy as np


def _add_src_to_path() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.append(str(root / "src"))
    sys.path.append(str(root / "src" / "models"))


_add_src_to_path()

from transformer_block import TransformerBlock, TransformerBlockConfig  # noqa: E402


def test_transformer_block_numpy() -> None:
    cfg = TransformerBlockConfig(dim=4, use_complex_attention=False)
    block = TransformerBlock(cfg)
    x = np.random.randn(2, 4)
    out = block(x)
    assert out.shape == (2, 4)


def test_transformer_block_complex_attention() -> None:
    cfg = TransformerBlockConfig(dim=4, use_complex_attention=True)
    block = TransformerBlock(cfg)
    x = np.random.randn(2, 4)
    out = block(x)
    assert out.shape == (2, 4)
