
import torch
from torch import nn
from torch.nn import functional as F

from reward_models.base import BaseModel


class AtariCnnRewardModel(BaseModel):
    def __init__(self, env):
        assert env.observation_space.shape == (4, 84, 84, 1), \
            f"Invalid input shape for reward model: " \
            f"Input shape {env.observation_space.shape} but expected (4, 84, 84, 1). " \
            f"Use this reward model only for Atari environments with screen size 84x84 (or compatible environments)."
        super().__init__(env)
        self.conv1 = nn.Conv2d(4, 16, kernel_size=7, stride=3)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(p=0.5)

        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, stride=2)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.dropout2 = nn.Dropout(p=0.5)

        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.batchnorm3 = nn.BatchNorm2d(16)
        self.dropout3 = nn.Dropout(p=0.5)

        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.batchnorm4 = nn.BatchNorm2d(16)
        self.dropout4 = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, observation):
        observation = observation.reshape(-1, 4, 1, 84, 84)
        observation = observation.reshape(-1, 4, 84, 84)
        observation = observation.type(torch.float32)

        x = F.leaky_relu(self.batchnorm1(self.conv1(observation)), 0.01)
        x = self.dropout1(x)

        x = F.leaky_relu(self.batchnorm2(self.conv2(x)), 0.01)
        x = self.dropout2(x)

        x = F.leaky_relu(self.batchnorm3(self.conv3(x)), 0.01)
        x = self.dropout3(x)

        x = F.leaky_relu(self.batchnorm4(self.conv4(x)), 0.01)
        x = self.dropout4(x)

        x = x.reshape(-1, 16 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

