from unittest.mock import Mock

import gym
import numpy as np
import torch

from data_generation.experience import Experience
from reward_modeling.utils import Preprocessor


def test_prepare_data(env):
    preprocessor = Preprocessor(env, num_stacked_frames=2)

    observation1 = np.array([(111., 112., 113.), (121., 122., 123.)])
    action1 = 1
    experience1 = Experience(observation=observation1, action=action1)

    observation2 = np.array([(211., 212., 213.), (221., 222., 223.)])
    action2 = 2
    experience2 = Experience(observation=observation2, action=action2)

    prediction_buffer = [experience1, experience2]

    preprocessor.env.observation_space.shape = tuple([2, 3])
    preprocessor.env.action_space.shape = tuple()

    prepared_data = preprocessor.prepare_data(prediction_buffer)

    assert np.array_equal(prepared_data, np.hstack([observation1.ravel(), action1, observation2.ravel(), action2]))


def test_combine_arrays(preprocessor):
    observation = np.array([1, 2, 3, 4])
    action = 5

    flattened_step_info = preprocessor.combine_arrays(observation, action)

    assert torch.all(torch.eq(flattened_step_info, torch.tensor([1, 2, 3, 4, 5])))


def test_convert_scalar_action_to_array(preprocessor):
    action = 1
    action_space_shape = tuple()

    preprocessor.env = Mock(name="environment", spec_set=gym.Env)
    preprocessor.env.action_space.shape = action_space_shape

    action_array = preprocessor.convert_action_to_array(action=action)

    assert np.array_equal(action_array, 1)


def test_convert_multidimensional_action_to_array(preprocessor):
    action_space_shape = tuple([2])
    action = np.array([1, 2])

    preprocessor.env = Mock(name="environment", spec_set=gym.Env)
    preprocessor.env.action_space.shape = action_space_shape

    action_array = preprocessor.convert_action_to_array(action=action)

    assert np.array_equal(action_array, np.array([1, 2]))


def test_convert_scalar_observation_to_array(preprocessor):
    observation = 1
    observation_space_shape = tuple()

    preprocessor.env = Mock(name="environment", spec_set=gym.Env)
    preprocessor.env.observation_space.shape = observation_space_shape

    observation_array = preprocessor.convert_observation_to_array(observation=observation)

    assert np.array_equal(observation_array, 1)


def test_convert_multidimensional_observation_to_array(preprocessor):
    observation = np.array([1, 2])
    observation_space_shape = tuple([2])

    preprocessor.env = Mock(name="environment", spec_set=gym.Env)
    preprocessor.env.observation_space.shape = observation_space_shape

    observation_array = preprocessor.convert_observation_to_array(observation=observation)

    assert np.array_equal(observation_array, np.array([1, 2]))


def test_convert_experience_to_array(preprocessor):
    experience = Experience(observation=np.array([3, 2]), action=1)
    experience_array = preprocessor.convert_experience_to_array(experience)

    assert torch.all(torch.eq(experience_array, torch.tensor([3, 2, 1])))


def test_add_first_experience(preprocessor):
    preprocessor.env = Mock(name="environment", spec_set=gym.Env)
    preprocessor.env.action_space.shape = tuple()
    preprocessor.env.observation_space.shape = tuple([2])

    experience = np.array([1, 2, 3])

    data = preprocessor.add_experience(data=np.zeros(6), experience_array=experience, i=0)

    assert np.array_equal(data, np.array([0, 0, 0, 1, 2, 3]))


def test_add_subsequent_experience(preprocessor):
    preprocessor.env = Mock(name="environment", spec_set=gym.Env)
    preprocessor.env.action_space.shape = tuple()
    preprocessor.env.observation_space.shape = tuple([2])

    experience = np.array([1, 2, 3])

    data = preprocessor.add_experience(data=np.zeros(6), experience_array=experience, i=1)

    assert np.array_equal(data, np.array([1, 2, 3, 0, 0, 0]))
