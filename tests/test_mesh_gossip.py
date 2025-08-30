import asyncio

import pytest

from pro_mesh import MeshNode


@pytest.mark.asyncio
async def test_two_nodes_converge() -> None:
    key = b"secret"
    node1 = MeshNode(key=key)
    node2 = MeshNode(key=key)
    await node1.start()
    await node2.start()
    node1.join("127.0.0.1", node2.port)
    node2.join("127.0.0.1", node1.port)
    node1.update_adapter("foo", {"a": 1.0}, 1)
    await node1.gossip()
    await asyncio.sleep(0.1)
    assert node2.adapters["foo"]["version"] == 1

    # conflicting updates with same version; higher metric should win
    node1.update_adapter("foo", {"a": 1.1}, 2)
    node2.update_adapter("foo", {"a": 1.2}, 2)
    await node1.gossip()
    await node2.gossip()
    await asyncio.sleep(0.1)
    assert node1.adapters["foo"]["weights"] == {"a": pytest.approx(1.2)}
    assert node2.adapters["foo"]["weights"] == {"a": pytest.approx(1.2)}

    await node1.leave()
    await node2.leave()


@pytest.mark.asyncio
async def test_health_check() -> None:
    key = b"secret"
    node = MeshNode(key=key)
    await node.start()
    ok = await node.health_check("127.0.0.1", node.port)
    assert ok
    await node.leave()
