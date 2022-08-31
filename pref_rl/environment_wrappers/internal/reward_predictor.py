import numpy as np
import torch
from gym import Wrapper

from ..info_dict_keys import PENALIZED_TRUE_REW


class RewardPredictor(Wrapper):
    def __init__(self, env, reward_model):
        super().__init__(env)
        self.reward_model = reward_model
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.reward_model.cuda()
        else:
            self.device = torch.device('cpu')

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        info[PENALIZED_TRUE_REW] = reward

        return observation, self.reward(observation), done, info

    def reward(self, observation):
        input_data = self._prepare_for_model(observation)
        # TODO: activate evaluation mode for model,
        #  see https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=train#torch.nn.Module.train
        return float(self.reward_model(input_data))

    def _prepare_for_model(self, observation):
        return torch.unsqueeze(torch.as_tensor(np.array(observation), device=self.device), dim=0)
