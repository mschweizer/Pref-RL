import random

import numpy as np
import pytest
import torch

from ...reward_modeling.atari_cnn import AtariCnnRewardModel


def test_has_correct_input_dimension(pong_env):
    reward_model = AtariCnnRewardModel(pong_env)

    assert reward_model.conv1.in_channels == pong_env.observation_space.shape[0]


def test_forward_pass(pong_env):
    reward_model = AtariCnnRewardModel(pong_env)

    action = random.randint(0, (pong_env.action_space.n - 1))
    pong_env.reset()
    obs = pong_env.step(action)[0]

    observation = torch.unsqueeze(torch.as_tensor(np.array(obs)), dim=0)
    prediction = reward_model(observation)

    assert prediction is not None


def test_throws_error_for_wrong_input_dims(cartpole_env):
    with pytest.raises(AssertionError) as exception_info:
        AtariCnnRewardModel(cartpole_env)
    assert isinstance(exception_info.value, AssertionError)
    assert exception_info.value.args[0] == "Invalid input shape for reward model: " \
                                           "Input shape (4, 4) but expected (4, 84, 84, 1). " \
                                           "Use this reward model only for Atari environments with screen size 84x84 " \
                                           "(or compatible environments)."
