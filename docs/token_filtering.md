# Token Filtering API

The `morphology` module provides a helper to select tokens based on grammar
information. The function returns the indices of tokens whose tags satisfy the
specified inclusion and exclusion sets.

```python
from morphology import filter_by_tags

tokens = ["я", "иду", "домой"]
tags = ["PRON", "VERB", "NOUN"]
idx = filter_by_tags(tokens, tags, include={"VERB", "NOUN"})
```

These indices can be converted into a boolean mask and passed to the
`wave_attention` function to restrict attention to the chosen tokens:

```python
import numpy as np
from transformers.blocks.attention import wave_attention

mask = np.zeros(len(tokens), dtype=bool)
mask[idx] = True
query = np.random.randn(len(tokens), 8).astype(np.complex64)
key = np.random.randn(len(tokens), 8).astype(np.complex64)
value = np.random.randn(len(tokens), 8).astype(np.complex64)

out = wave_attention(query, key, value, mask=mask)
```

Passing the mask ensures that the matrix multiplication inside the attention
mechanism only considers the filtered tokens, reducing computation while
preserving output quality.
