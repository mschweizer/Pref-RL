from collections import deque

import gym
import numpy as np


# TODO: Use VecNormalize Wrapper instead?
class RewardStandardizer(gym.RewardWrapper):
    def __init__(self, env, desired_std=.05, update_interval=30000, buffer_size=3000):
        super().__init__(env)
        self.update_interval = update_interval
        self.buffer = deque(maxlen=buffer_size)
        self.counter = 0
        self.update_interval = update_interval
        self.mean = None
        self.std = None
        self.desired_std = desired_std

    def reward(self, reward):
        self.buffer.append(reward)
        if self.counter % self.update_interval == 0:
            self._set_standardization_params()
        self.counter += 1
        return self._standardize(reward)

    def _set_standardization_params(self):
        numpy_buffer = np.array(self.buffer)
        self.mean = numpy_buffer.mean()
        self.std = numpy_buffer.std()

    def _standardize(self, reward):
        assert self.mean is not None and self.std is not None, "Reward standardization parameters have not been set."
        if self.std > 0:
            return (reward - self.mean) / (self.std / self.desired_std)
        else:
            return reward - self.mean
