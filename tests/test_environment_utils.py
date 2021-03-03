import gym
import numpy as np
from stable_baselines3.common.atari_wrappers import AtariWrapper

from environment.utils import wrap_env, create_env, is_atari_env, is_wrapped


def test_converts_to_stacked_env():
    env = gym.make('CartPole-v1')
    frame_stack_depth = 5
    shp = env.observation_space.shape
    env = wrap_env(env, frame_stack_depth)

    assert len(env.observation_space.shape) == len(np.hstack([frame_stack_depth, shp]))
    assert np.all(env.observation_space.shape == np.hstack([frame_stack_depth, shp]))


def test_is_atari_env():
    cartpole_env = create_env('CartPole-v1')
    pong_env = create_env('Pong-v0')

    assert not is_atari_env(cartpole_env)
    assert is_atari_env(pong_env)


def test_is_wrapped():
    cartpole_env = create_env('CartPole-v1')
    pong_env = create_env('Pong-v0')

    assert not is_wrapped(cartpole_env, AtariWrapper)
    assert is_wrapped(pong_env, AtariWrapper)


def test_adds_atari_wrapper():
    env = create_env('Pong-v0')
    assert is_wrapped(env, AtariWrapper)
