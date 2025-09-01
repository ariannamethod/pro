import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from resonance.hypergraph import HyperGraph


def test_hypergraph_load_benchmark():
    g = HyperGraph()
    start = time.perf_counter()
    for i in range(1000):
        g.add_node(f"n{i}", {"content": str(i)}, connect=[f"n{i-1}"] if i else None)
    duration = time.perf_counter() - start
    assert g.get_node("n500") is not None
    assert duration < 1.0
