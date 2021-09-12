import numpy as np

from models.reward.mlp import MlpRewardModel


def test_has_correct_input_dimension(env):
    reward_model = MlpRewardModel(env)

    assert reward_model.fc1.in_features == np.prod(env.observation_space.shape)
