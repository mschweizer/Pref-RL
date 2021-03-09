from unittest.mock import Mock

import pytest
from stable_baselines3 import A2C

from data_generation.preference_data_generator import PreferenceDataGenerator


@pytest.fixture()
def policy_model_with_reward_wrapper(reward_wrapper):
    return A2C('MlpPolicy', env=reward_wrapper, n_steps=10)


@pytest.fixture()
def preference_data_generator(policy_model_with_reward_wrapper):
    return PreferenceDataGenerator(policy_model=policy_model_with_reward_wrapper, segment_length=3)


@pytest.mark.parametrize('training', [True, False])
def test_generates_k_preferences(preference_data_generator, training):
    if not training:
        preference_data_generator.policy_model.learn = Mock()

    generation_volume = 1
    preferences = preference_data_generator.generate(generation_volume=generation_volume, with_training=training)
    if not training:
        preference_data_generator.policy_model.learn.assert_not_called()

    assert len(preferences) == generation_volume
