"""Deprecated wrapper for memory.pooling."""

import sys
import types
import warnings

from memory import pooling as _pool

warnings.warn(
    "pro_memory_pool is deprecated; use memory.pooling instead",
    DeprecationWarning,
    stacklevel=2,
)


class _Proxy(types.ModuleType):
    def __getattr__(self, name):
        return getattr(_pool, name)

    def __setattr__(self, name, value):
        setattr(_pool, name, value)


module = sys.modules[__name__]
module.__class__ = _Proxy
