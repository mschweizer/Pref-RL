import time

import gym
from stable_baselines3.common.monitor import Monitor


class RewardMonitor(Monitor):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.original_rewards = None

    def reset(self, **kwargs):
        self.original_rewards = []
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        original_reward = info['original_reward']
        self.original_rewards.append(original_reward)
        if info['original_done']:
            ep_rew = sum(self.original_rewards)
            ep_len = len(self.original_rewards)
            ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
            self.original_rewards = []
            info["episode"] = ep_info

        return observation, reward, done, info
