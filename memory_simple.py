"""Простая замена MemoryStore для Railway"""

import numpy as np
from typing import List, Optional, Dict, Any

class MemoryStore:
    """Минимальная реализация MemoryStore для совместимости"""
    
    def __init__(self, dim: int = 32):
        self.dim = dim
        self.embeddings = np.array([]).reshape(0, dim)
        self.contents: List[str] = []
    
    def add_utterance(self, dialog_id: str, role: str, content: str, embedding: Optional[np.ndarray] = None):
        """Добавить сообщение"""
        if embedding is not None:
            if len(self.embeddings) == 0:
                self.embeddings = embedding.reshape(1, -1)
            else:
                self.embeddings = np.vstack([self.embeddings, embedding.reshape(1, -1)])
            self.contents.append(content)
    
    def most_similar(self, query_vec: np.ndarray, top_k: int = 5, dialog_id: str = "global") -> List[str]:
        """Найти похожие"""
        if len(self.embeddings) == 0:
            return []
        
        distances = np.linalg.norm(self.embeddings - query_vec.reshape(1, -1), axis=1)
        indices = np.argsort(distances)[:top_k]
        return [self.contents[i] for i in indices if i < len(self.contents)]


class GraphRetriever:
    """Заглушка для GraphRetriever"""
    def __init__(self, store):
        pass


class ReinforceRetriever:
    """Заглушка для ReinforceRetriever"""
    def __init__(self, store, dim: int = 32, lr: float = 0.1):
        pass


def cache_layers(*args, **kwargs):
    """Заглушка для cache_layers"""
    pass
