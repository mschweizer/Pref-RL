from unittest.mock import Mock

from gym import Env

from wrappers.internal.reward_monitor import RewardMonitor


def test_info_does_not_change_when_not_done():
    info = {'original_done': False, 'original_reward': 10}
    feedback_removed_env = Mock(spec_set=Env, step=Mock(return_value=("obs", "reward", "done", info)))
    monitored_env = RewardMonitor(feedback_removed_env)
    monitored_env.reset()
    obs, rew, done, new_info = monitored_env.step(monitored_env.action_space.sample())
    assert new_info == info
    assert monitored_env.original_rewards[-1] == info['original_reward']


def test_proper_update_of_info():
    info = {'original_done': True, 'original_reward': 10}
    feedback_removed_env = Mock(spec_set=Env, step=Mock(return_value=("obs", "reward", "done", info)))
    monitored_env = RewardMonitor(feedback_removed_env)
    monitored_env.reset()

    monitored_env.original_rewards = [10, 15, 20, 15, 0]
    sum_of_rewards = sum(monitored_env.original_rewards)
    obs, rew, done, info = monitored_env.step(monitored_env.action_space.sample())

    assert info['episode']['r'] == round((sum_of_rewards + info['original_reward']), 6)
