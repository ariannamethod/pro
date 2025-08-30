import re
from typing import Dict

import numpy as np


class SymbolicAnd:
    """Element-wise logical AND layer for binary arrays."""

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.logical_and(a > 0, b > 0).astype(np.float32)


class SymbolicOr:
    """Element-wise logical OR layer for binary arrays."""

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.logical_or(a > 0, b > 0).astype(np.float32)


class SymbolicNot:
    """Element-wise logical NOT layer for binary arrays."""

    def __call__(self, a: np.ndarray) -> np.ndarray:
        return np.logical_not(a > 0).astype(np.float32)


class SymbolicReasoner:
    """Evaluate simple boolean expressions with symbolic layers.

    The reasoner supports the operators ``AND``, ``OR`` and ``NOT``.  Variables
    in *expr* are looked up in *facts* and assumed ``False`` if missing.
    """

    def __init__(self) -> None:
        self.and_layer = SymbolicAnd()
        self.or_layer = SymbolicOr()
        self.not_layer = SymbolicNot()

    def evaluate(self, expr: str, facts: Dict[str, bool]) -> bool:
        """Evaluate *expr* using truth assignments from *facts*.

        Examples
        --------
        >>> r = SymbolicReasoner()
        >>> r.evaluate("A AND NOT B", {"A": True, "B": False})
        True
        """

        def repl(match: re.Match[str]) -> str:
            var = match.group(0)
            if var in {"AND", "OR", "NOT"}:
                return var
            return str(bool(facts.get(var, False)))

        safe = re.sub(r"[A-Za-z_]+", repl, expr)
        safe = safe.replace("AND", "and").replace("OR", "or").replace("NOT", "not")
        return bool(eval(safe, {}, {}))
