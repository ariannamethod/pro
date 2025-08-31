"""Deprecated wrapper for memory.storage and memory.lattice."""

import sys
import types
import warnings

from memory import storage as _storage
from memory import lattice as _lattice

warnings.warn(
    "pro_memory is deprecated; use memory.storage instead",
    DeprecationWarning,
    stacklevel=2,
)


class _Proxy(types.ModuleType):
    def __getattr__(self, name):
        if hasattr(_storage, name):
            return getattr(_storage, name)
        if hasattr(_lattice, name):
            return getattr(_lattice, name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if hasattr(_storage, name):
            setattr(_storage, name, value)
        elif hasattr(_lattice, name):
            setattr(_lattice, name, value)
        else:
            super().__setattr__(name, value)


module = sys.modules[__name__]
module.__class__ = _Proxy
