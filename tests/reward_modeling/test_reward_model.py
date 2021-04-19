import numpy as np

from reward_modeling.models.reward import RewardModel


def test_has_correct_input_dimension(env):
    reward_model = RewardModel(env)

    assert reward_model.fc1.in_features == np.prod(env.observation_space.shape)
