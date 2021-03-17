from collections import deque

import numpy as np
import torch
from gym import Wrapper

from preference_data.preference.experience import Experience


# TODO: Use gym.core.RewardWrapper instead of custom reward_modeling wrapper
class RewardPredictor(Wrapper):
    def __init__(self, env, reward_model, trajectory_buffer_size):
        super().__init__(env)
        self.trajectory_buffer = deque(maxlen=trajectory_buffer_size)
        self.reward_model = reward_model
        self._last_observation = None
        self._last_done = False

    def reset(self, **kwargs):
        self._last_observation = super().reset(**kwargs)
        self._last_done = False
        return self._last_observation

    def step(self, action):
        new_observation, reward, new_done, info = super().step(action)

        # A reward_modeling tensor is explicitly created because stable baselines performs a deep copy on 'info'
        # Torch otherwise throws a 'RuntimeError: Only Tensors created explicitly by the user (graph leaves)
        # support the deepcopy protocol at the moment'
        info['original_reward'] = torch.tensor(reward)

        # TODO: should this really be the last observation / done? see implementation of stable baselines
        transformed_reward = self.reward(self._last_observation)
        experience = Experience(self._last_observation, action, transformed_reward, self._last_done, info)

        self.trajectory_buffer.append(experience)

        self._last_observation = new_observation
        self._last_done = new_done

        return new_observation, transformed_reward, new_done, info

    def reward(self, observation):
        input_data = self._prepare_for_model(observation)
        return self.reward_model(input_data)

    @staticmethod
    def _prepare_for_model(observation):
        return torch.as_tensor([np.array(observation)])
