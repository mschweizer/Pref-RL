import torch
from torch import nn
from torch.nn import functional as F


class RewardModel(nn.Module):
    def __init__(self, input_dimension):
        super().__init__()

        self.fc1 = nn.Linear(input_dimension, 64)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, 64)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class ChoiceModel(nn.Module):
    def __init__(self, reward_model):
        super().__init__()
        self.reward_model = reward_model

    def forward(self, query):
        total_rewards = self.sum_segment_rewards(query)
        return self.compute_choice_probability(total_rewards)

    def sum_segment_rewards(self, query):
        rewards = self.reward_model(query.reshape(-1, 20))
        original_shape = query.shape[:3]  # only first three dimensions of original shape because rewards are scalar
        return rewards.reshape(original_shape).sum(axis=2)

    @staticmethod
    def compute_choice_probability(total_rewards):
        return F.softmax(total_rewards)[:, 0].type(torch.float64)
