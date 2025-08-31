import asyncio
import pytest

from trainer import Trainer
from resonance.p2p_resonance import P2PResonance


@pytest.mark.asyncio
async def test_two_trainers_sync() -> None:
    peer1 = P2PResonance()
    peer2 = P2PResonance()
    await peer1.start()
    await peer2.start()
    t1 = Trainer(resonance_peer=peer1)
    t2 = Trainer(resonance_peer=peer2)
    await asyncio.to_thread(t1.train_step, "w", 1.0)
    await asyncio.to_thread(t2.train_step, "w", 2.0)
    await peer1.exchange(*peer2.server_address)
    assert t1.params["w"] == pytest.approx(3.0)
    assert t2.params["w"] == pytest.approx(3.0)
    await peer1.stop()
    await peer2.stop()
