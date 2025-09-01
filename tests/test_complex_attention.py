import pathlib
import sys

import numpy as np


def _add_src_to_path() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    src = root / "src"
    sys.path.append(str(src))


_add_src_to_path()

from attention import ComplexAttention  # noqa: E402


def test_complex_attention_runs() -> None:
    attn = ComplexAttention(4)
    x = np.random.randn(2, 4)
    out = attn(x)
    out_zero = attn(x, resistance=np.zeros(x.shape[0]))
    assert isinstance(out_zero, np.ndarray)
    assert out_zero.shape == (2, 4)
    assert np.allclose(out, out_zero)
