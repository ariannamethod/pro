import pytest

from trainer import Trainer
from resonance.p2p_resonance import P2PResonance


def test_two_trainers_sync() -> None:
    peer1 = P2PResonance()
    peer2 = P2PResonance()
    t1 = Trainer(resonance_peer=peer1)
    t2 = Trainer(resonance_peer=peer2)
    t1.train_step("w", 1.0)
    t2.train_step("w", 2.0)
    peer1.exchange("127.0.0.1", peer2.server_address[1])
    assert t1.params["w"] == pytest.approx(3.0)
    assert t2.params["w"] == pytest.approx(3.0)
    peer1.stop()
    peer2.stop()
