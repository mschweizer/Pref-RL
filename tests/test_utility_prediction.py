from unittest.mock import Mock, patch

import gym
import numpy as np
import torch

from experience import Experience, ExperienceBuffer, PredictionBuffer
from reward_predictor import RewardPredictor


def test_predict_utility(env, reward_predictor):
    num_stacked_frames = 4
    prediction_buffer = ExperienceBuffer(size=num_stacked_frames)

    env.reset()

    for i in range(num_stacked_frames):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        prediction_buffer.append(Experience(observation, action, reward, done, info))

    predicted_utility = reward_predictor.predict_utility()

    assert predicted_utility.dtype == torch.float32


def test_prepare_data(reward_predictor):
    observation1 = np.array([(111., 112., 113.), (121., 122., 123.)])
    action1 = 1
    experience1 = Experience(observation=observation1, action=action1)

    observation2 = np.array([(211., 212., 213.), (221., 222., 223.)])
    action2 = 2
    experience2 = Experience(observation=observation2, action=action2)

    reward_predictor.prediction_buffer = [experience1, experience2]
    reward_predictor.environment.observation_space.shape = tuple([2, 3])
    reward_predictor.environment.action_space.shape = tuple()

    prepared_data = reward_predictor.prepare_data()

    assert np.array_equal(prepared_data, np.hstack([observation1.ravel(), action1, observation2.ravel(), action2]))


def test_get_flattened_lengths():
    env = Mock()
    env.action_space.shape = ()
    env.observation_space.shape = 4

    with patch('reward_predictor.RewardNet'):
        reward_predictor = RewardPredictor(env=env,
                                           trajectory_buffer=PredictionBuffer(size=2, prediction_stack_depth=4),
                                           num_stacked_frames=2,
                                           training_interval=10)
        assert reward_predictor.get_flattened_action_space_length() == 1
        assert reward_predictor.get_flattened_observation_space_length() == 4
        assert reward_predictor.get_flattened_experience_length() == 5
        assert reward_predictor.get_flattened_input_length() == 10


def test_combine_arrays(reward_predictor):
    observation = np.array([1, 2, 3, 4])
    action = 5

    flattened_step_info = reward_predictor.combine_arrays(observation, action)

    assert torch.all(torch.eq(flattened_step_info, torch.tensor([1, 2, 3, 4, 5])))


def test_convert_scalar_action_to_array(reward_predictor):
    action = 1
    action_space_shape = tuple()

    reward_predictor.environment = Mock(name="environment", spec_set=gym.Env)
    reward_predictor.environment.action_space.shape = action_space_shape

    action_array = reward_predictor.convert_action_to_array(action=action)

    assert np.array_equal(action_array, 1)


def test_convert_multidimensional_action_to_array(reward_predictor):
    action_space_shape = tuple([2])
    action = np.array([1, 2])

    reward_predictor.environment = Mock(name="environment", spec_set=gym.Env)
    reward_predictor.environment.action_space.shape = action_space_shape

    action_array = reward_predictor.convert_action_to_array(action=action)

    assert np.array_equal(action_array, np.array([1, 2]))


def test_convert_scalar_observation_to_array(reward_predictor):
    observation = 1
    observation_space_shape = tuple()

    reward_predictor.environment = Mock(name="environment", spec_set=gym.Env)
    reward_predictor.environment.observation_space.shape = observation_space_shape

    observation_array = reward_predictor.convert_observation_to_array(observation=observation)

    assert np.array_equal(observation_array, 1)


def test_convert_multidimensional_observation_to_array(reward_predictor):
    observation = np.array([1, 2])
    observation_space_shape = tuple([2])

    reward_predictor.environment = Mock(name="environment", spec_set=gym.Env)
    reward_predictor.environment.observation_space.shape = observation_space_shape

    observation_array = reward_predictor.convert_observation_to_array(observation=observation)

    assert np.array_equal(observation_array, np.array([1, 2]))


def test_convert_experience_to_array(reward_predictor):
    experience = Experience(observation=np.array([3, 2]), action=1)
    experience_array = reward_predictor.convert_experience_to_array(experience)

    assert torch.all(torch.eq(experience_array, torch.tensor([3, 2, 1])))


def test_add_first_experience(reward_predictor):
    reward_predictor.environment = Mock(name="environment", spec_set=gym.Env)
    reward_predictor.environment.action_space.shape = tuple()
    reward_predictor.environment.observation_space.shape = tuple([2])

    experience = np.array([1, 2, 3])

    data = reward_predictor.add_experience(data=np.zeros(6), experience_array=experience, i=0)

    assert np.array_equal(data, np.array([0, 0, 0, 1, 2, 3]))


def test_add_subsequent_experience(reward_predictor):
    reward_predictor.environment = Mock(name="environment", spec_set=gym.Env)
    reward_predictor.environment.action_space.shape = tuple()
    reward_predictor.environment.observation_space.shape = tuple([2])

    experience = np.array([1, 2, 3])

    data = reward_predictor.add_experience(data=np.zeros(6), experience_array=experience, i=1)

    assert np.array_equal(data, np.array([1, 2, 3, 0, 0, 0]))

# def test_training_changes_model_parameters(reward_training_env):
#     """
#     Testing whether parameters change uses code from  / is based on
#     https://github.com/suriyadeepan/torchtest/blob/66a2c8b669aa23601f64e208463e9449ffc135da/torchtest/torchtest.py#L106
#     """
#
#     model = reward_training_env.reward_predictor.utility_model
#     params = [np for np in model.named_parameters() if np[1].requires_grad]
#     initial_params = [(name, p.clone()) for (name, p) in params]
#
#     reward_training_env.train_reward_model()
#
#     for (_, p0), (name, p1) in zip(initial_params, params):
#         assert not torch.equal(p0, p1)
