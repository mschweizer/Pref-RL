from unittest.mock import Mock

from gym import Env

from environment.never_done_wrapper import NoIndirectFeedbackWrapper
from environment.utils import create_env


def test_is_never_done():
    env = Mock(spec_set=Env, step=Mock(return_value=("obs", 1., True, None)))
    wrapped_env = NoIndirectFeedbackWrapper(env)

    wrapped_env.reset()
    action = env.action_space.sample()
    _, _, done, _ = wrapped_env.step(action)

    assert done is False


def test_applies_penalty_when_episode_ends():
    reward = 1
    penalty = 3
    done = True

    env = Mock(spec_set=Env, step=Mock(return_value=("obs", reward, done, None)))
    wrapped_env = NoIndirectFeedbackWrapper(env, penalty)

    wrapped_env.reset()
    action = env.action_space.sample()
    _, penalized_reward, _, _ = wrapped_env.step(action)

    assert penalized_reward == reward - penalty


def test_does_not_apply_penalty_when_episode_continues():
    reward = 1
    done = False

    env = Mock(spec_set=Env, step=Mock(return_value=("obs", reward, done, None)))
    wrapped_env = NoIndirectFeedbackWrapper(env, penalty=3.)

    wrapped_env.reset()
    action = env.action_space.sample()
    _, rew, _, _ = wrapped_env.step(action)

    assert rew == reward


def test_resets_env_when_episode_ends():
    done = True

    env = Mock(spec_set=Env, step=Mock(return_value=("obs", 1., done, None)))
    wrapped_env = NoIndirectFeedbackWrapper(env)

    wrapped_env.reset()
    action = env.action_space.sample()
    wrapped_env.step(action)

    assert env.reset.call_count == 2


def test_does_not_reset_env_when_episode_continues():
    done = False

    env = Mock(spec_set=Env, step=Mock(return_value=("obs", 1., done, None)))
    wrapped_env = NoIndirectFeedbackWrapper(env)

    wrapped_env.reset()
    action = env.action_space.sample()
    wrapped_env.step(action)

    assert env.reset.call_count == 1


def test_does_not_include_life_info():
    atari_env = create_env('Pong-v0')
    wrapped_env = NoIndirectFeedbackWrapper(atari_env)

    wrapped_env.reset()
    action = atari_env.action_space.sample()
    _, _, _, info = wrapped_env.step(action)

    assert "ale.lives" not in info.keys()
