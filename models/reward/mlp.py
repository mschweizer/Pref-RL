import numpy as np
from torch import nn
from torch.nn import functional as F

from models.reward.base import BaseModel


class MlpRewardModel(BaseModel):
    def __init__(self, env):
        super().__init__(env)

        self.fc1 = nn.Linear(self._get_flattened_input_length(), 64)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, 64)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, observation):
        flattened_observation = self._flatten_observation(observation).float()
        x = F.leaky_relu(self.fc1(flattened_observation))
        x = self.dropout1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout2(x)
        reward_prediction = self.fc3(x)
        return reward_prediction

    @staticmethod
    def _flatten_observation(observation):
        return observation.reshape(observation.shape[0], -1)

    def _get_flattened_input_length(self):
        return int(np.prod(self.environment.observation_space.shape))
