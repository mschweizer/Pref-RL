import gym
import numpy as np
import pytest
from stable_baselines3.common.atari_wrappers import AtariWrapper

from environment.no_indirect_feedback_wrapper import NoIndirectFeedbackWrapper
from environment.utils import add_external_env_wrappers, create_env, is_atari_env, is_wrapped, add_internal_env_wrappers
from reward_modeling.reward_model import RewardModel
from reward_modeling.reward_standardization_wrapper import RewardStandardizationWrapper
from reward_modeling.reward_wrapper import RewardWrapper


@pytest.fixture()
def envs():
    cartpole_env = create_env('CartPole-v1', termination_penalty=0)
    pong_env = create_env('Pong-v0', termination_penalty=0)
    return cartpole_env, pong_env


def test_converts_to_stacked_env():
    env = gym.make('CartPole-v1')
    frame_stack_depth = 5
    shp = env.observation_space.shape
    env = add_external_env_wrappers(env, frame_stack_depth=frame_stack_depth, termination_penalty=0.)

    assert len(env.observation_space.shape) == len(np.hstack([frame_stack_depth, shp]))
    assert np.all(env.observation_space.shape == np.hstack([frame_stack_depth, shp]))


def test_is_atari_env(envs):
    cartpole_env, pong_env = envs
    assert not is_atari_env(cartpole_env)
    assert is_atari_env(pong_env)


def test_is_wrapped(envs):
    cartpole_env, pong_env = envs
    assert not is_wrapped(cartpole_env, AtariWrapper)
    assert is_wrapped(pong_env, AtariWrapper)


def test_wrap_external_environment(envs):
    _, pong_env = envs
    assert is_wrapped(pong_env, AtariWrapper)
    assert is_wrapped(pong_env, NoIndirectFeedbackWrapper)


def test_wrap_internal_environment(cartpole_env, ):
    reward_model = RewardModel(cartpole_env)

    wrapped_env = add_internal_env_wrappers(cartpole_env,
                                            reward_model=reward_model,
                                            desired_std=1.,
                                            trajectory_buffer_size=1,
                                            standardization_buffer_size=1,
                                            standardization_params_update_interval=1)

    assert is_wrapped(wrapped_env, RewardWrapper) and is_wrapped(wrapped_env, RewardStandardizationWrapper)
