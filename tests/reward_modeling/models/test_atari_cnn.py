import random

import numpy as np
import torch

from reward_modeling.models.reward.atari_cnn import AtariCnnRewardModel


def test_has_correct_input_dimension(pong_env):
    reward_model = AtariCnnRewardModel(pong_env)

    assert reward_model.conv1.in_channels == pong_env.observation_space.shape[0]


def test_forward_pass(pong_env):
    reward_model = AtariCnnRewardModel(pong_env)

    action = random.randint(0, (pong_env.action_space.n - 1))
    pong_env.reset()
    obs = pong_env.step(action)[0]

    observation = torch.as_tensor([np.array(obs)])
    prediction = reward_model(observation)

    assert prediction is not None
