import numpy as np
import torch
from gym import Wrapper


class RewardPredictor(Wrapper):
    def __init__(self, env, reward_model):
        super().__init__(env)
        self.reward_model = reward_model

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        info['external_reward'] = reward  # not necessarily equal to 'original_reward' due to termination penalty

        return observation, self.reward(observation), done, info

    def reward(self, observation):
        input_data = self._prepare_for_model(observation)
        return float(self.reward_model(input_data))

    @staticmethod
    def _prepare_for_model(observation):
        # TODO: converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
        #  (See Rob's mail)
        return torch.as_tensor([np.array(observation)])
