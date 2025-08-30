# Symbolic Reasoning Block

The `transformers.blocks.reasoning` module provides lightweight layers that
perform symbolic logic operations on numpy arrays. These layers are useful when
boolean structure needs to interact with neural components.

## Components

- **SymbolicAnd** – element-wise logical AND.
- **SymbolicOr** – element-wise logical OR.
- **SymbolicNot** – element-wise logical NOT.
- **SymbolicReasoner** – evaluates simple boolean expressions composed of the
  above operators.

Example:

```python
from transformers.blocks import SymbolicReasoner

reasoner = SymbolicReasoner()
result = reasoner.evaluate("A AND NOT B", {"A": True, "B": False})
assert result is True
```
