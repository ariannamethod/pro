import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from resonance.hypergraph import HyperGraph
from router.subgraph import select_context_subgraph


def test_select_context_subgraph():
    g = HyperGraph()
    g.add_node("a", {"content": "hello world"})
    g.add_node("b", {"content": "foo bar"}, connect=["a"])
    g.add_node("c", {"content": "world peace"}, connect=["b"])
    sub = select_context_subgraph(g, ["world"])
    contents = sorted(n.data["content"] for n in sub.nodes.values())
    assert contents == ["hello world", "world peace"]
