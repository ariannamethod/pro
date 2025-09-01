"""Minimal demonstration of weightless resonant routing and fractal memory."""

from quantum_memory import QuantumMemory
from router import ResonantRouter
import numpy as np


def main() -> None:
    # Fractal encoding of tokens
    mem = QuantumMemory()
    tokens = ["hello", "world"]
    for i, tok in enumerate(tokens):
        mem.store(i, tok)
    print("Fractal amplitudes for 'hello':", mem.retrieve(0))

    # Weightless resonant routing
    router = ResonantRouter(threshold=0.1)
    features = np.random.rand(4, 8)
    mask = router.route(features)
    print("Resonant routing mask:", mask)


if __name__ == "__main__":
    main()
