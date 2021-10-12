import numpy as np
import torch
from gym import Wrapper


class RewardPredictor(Wrapper):
    def __init__(self, env, reward_model):
        super().__init__(env)
        self.reward_model = reward_model

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        # A models tensor is explicitly created because stable baselines performs a deep copy on 'info'
        # Torch otherwise throws a 'RuntimeError: Only Tensors created explicitly by the user (graph leaves)
        # support the deepcopy protocol at the moment'
        info['external_reward'] = torch.tensor(reward)

        return observation, self.reward(observation), done, info

    def reward(self, observation):
        input_data = self._prepare_for_model(observation)
        return float(self.reward_model(input_data))

    @staticmethod
    def _prepare_for_model(observation):
        return torch.as_tensor([np.array(observation)])
