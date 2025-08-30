import numpy as np

from router.policy import PatchRoutingPolicy


def test_policy_learns_to_use_quantum():
    policy = PatchRoutingPolicy(dim=2, lr=0.5, seed=0)
    features = np.ones((10, 2))
    baseline = 10.0  # classical energy cost
    for _ in range(100):
        mask = policy.route(features)
        energy = np.where(mask, 0.5, 1.0).mean()
        reward = -energy
        policy.update(reward)
    mask = policy.route(features)
    energy = np.where(mask, 0.5, 1.0).sum()
    assert energy < baseline
