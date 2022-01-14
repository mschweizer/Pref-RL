from unittest.mock import Mock

import numpy as np

from ....environment_wrappers.internal.reward_standardizer import RewardStandardizer


def test_buffers_rewards(cartpole_env):
    reward = 42.
    cartpole_env.step = Mock(return_value=("obs", reward, "done", "info"))
    standardization_wrapper = RewardStandardizer(cartpole_env)

    action = cartpole_env.action_space.sample()
    standardization_wrapper.step(action)

    assert standardization_wrapper.buffer[0] == reward


def test_sets_params(cartpole_env):
    reward = 42.
    cartpole_env.step = Mock(return_value=("obs", reward, "done", "info"))
    standardization_wrapper = RewardStandardizer(cartpole_env)

    standardization_wrapper.reset()
    action = standardization_wrapper.action_space.sample()
    standardization_wrapper.step(action)

    assert standardization_wrapper.mean == reward and standardization_wrapper.std == 0.


def test_standardizes_initial_reward(cartpole_env):
    reward = 42.

    cartpole_env.step = Mock(return_value=("obs", reward, "done", "info"))
    standardization_wrapper = RewardStandardizer(cartpole_env)

    standardization_wrapper.reset()
    action = standardization_wrapper.action_space.sample()
    _, normalized_rew, _, _ = standardization_wrapper.step(action)

    assert normalized_rew == 0.


def test_standardizes_reward():
    desired_std = 0.05
    update_interval = 20

    buffer = [14., 6., 6.]
    old_mean = np.array(buffer).mean()
    old_std = np.array(buffer).std()

    next_reward = 14.
    step_function_mock = Mock(return_value=("obs", next_reward, "done", "info"))
    standardization_wrapper = RewardStandardizer(env=Mock(step=step_function_mock),
                                                 desired_std=desired_std,
                                                 update_interval=update_interval)

    standardization_wrapper.buffer = buffer.copy()
    standardization_wrapper.counter = update_interval
    standardization_wrapper.mean = old_mean
    standardization_wrapper.std = old_std

    standardization_wrapper.reset()
    action = standardization_wrapper.action_space.sample()
    _, standardized_rew, _, _ = standardization_wrapper.step(action)

    buffer.append(next_reward)
    correct_mean = np.array(buffer).mean()
    correct_std = np.array(buffer).std()

    assert correct_mean == 10
    assert correct_std == 4

    assert standardized_rew == (next_reward - correct_mean) / (correct_std / desired_std)
