import numpy as np
from transformers.modeling_transformer import block_factory


def test_block_factory_generates_distinct_outputs():
    x = np.array([1.0, 1.0], dtype=np.float32)
    context = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    static_block = block_factory(2, 2, use_hyper=False)
    hyper_block = block_factory(2, 2, use_hyper=True)

    static_out = static_block(x, context)
    hyper_out = hyper_block(x, context)

    assert not np.allclose(static_out, hyper_out)
