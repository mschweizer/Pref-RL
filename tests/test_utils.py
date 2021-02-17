from unittest.mock import Mock

from reward_modeling.utils import get_flattened_action_space_length, \
    get_flattened_observation_space_length, get_flattened_experience_length, get_flattened_input_length


def test_get_flattened_lengths():
    env = Mock()
    env.action_space.shape = ()
    env.observation_space.shape = 4

    num_stacked_frames = 2

    assert get_flattened_action_space_length(env) == 1
    assert get_flattened_observation_space_length(env) == 4
    assert get_flattened_experience_length(env) == 5
    assert get_flattened_input_length(num_stacked_frames=num_stacked_frames, env=env) == 10
