import pathlib
import sys

import pytest


def _add_src_to_path():
    root = pathlib.Path(__file__).resolve().parents[1]
    src = root / "src"
    sys.path.append(str(src))


_add_src_to_path()

from models.transformer_block import TransformerBlockConfig  # noqa: E402

try:  # pragma: no cover - optional dependency
    import torch
    from attention.complex_attention import ComplexAttention  # noqa: E402
    from models.transformer_block import TransformerBlock  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - handled in skip conditions
    torch = None
    ComplexAttention = None
    TransformerBlock = None


def test_config_serialisation_roundtrip():
    cfg = TransformerBlockConfig(dim=4, use_complex_attention=True)
    data = cfg.to_dict()
    assert data["use_complex_attention"] is True
    new_cfg = TransformerBlockConfig.from_dict(data)
    assert new_cfg == cfg


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_complex_attention_runs():
    attn = ComplexAttention(4)
    x = torch.randn(2, 4)
    out = attn(x)
    assert out.shape == (2, 4)


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_transformer_block_uses_attention():
    cfg = TransformerBlockConfig(dim=4, use_complex_attention=True)
    block = TransformerBlock(cfg)
    x = torch.randn(2, 4)
    out = block(x)
    assert out.shape == x.shape
