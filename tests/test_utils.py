import numpy as np

from reward_modeling.utils import get_flattened_input_length


def test_get_flattened_lengths(env):
    assert get_flattened_input_length(env=env) == np.prod(env.observation_space.shape)
