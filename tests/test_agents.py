import numpy as np

from a3c import agents


def test_discounted_rewards():
    rewards = np.ones(100)

    discounted_rewards = agents.discounted_rewards(rewards, 1.0)
    assert np.all(discounted_rewards == np.arange(100, 0, -1))

    discounted_rewards = agents.discounted_rewards(rewards, 0.0)
    assert np.all(discounted_rewards == rewards)

    discounted_rewards = agents.discounted_rewards(rewards, 0.5)
    assert discounted_rewards[-1] == 1
    assert discounted_rewards[-2] == 1.5
    assert discounted_rewards[-3] == 1.75
