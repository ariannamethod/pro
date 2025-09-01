"""Заглушка для ReinforceRetriever"""

from typing import Any
from .store import MemoryStore

class ReinforceRetriever:
    """Заглушка для ReinforceRetriever"""
    def __init__(self, store: MemoryStore, dim: int = 32, lr: float = 0.1):
        self.store = store
        self.dim = dim
        self.lr = lr
