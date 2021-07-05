import torch
from torch import nn
from torch.nn import functional as F


class ChoiceModel(nn.Module):
    def __init__(self, reward_model):
        super().__init__()
        self.reward_model = reward_model

    def forward(self, query):
        total_rewards = self.sum_segment_rewards(query)
        return self.compute_choice_probability(total_rewards)

    def sum_segment_rewards(self, query):
        num_reward_model_inputs = self._calculate_input_size(query)
        new_shape = (num_reward_model_inputs,) + tuple(query.shape[3:])
        rewards = self.reward_model(query.reshape(new_shape))
        original_shape = query.shape[:3]  # only first three dimensions of original shape because rewards are scalar
        return rewards.reshape(original_shape).sum(axis=2)

    @staticmethod
    def _calculate_input_size(query):
        num_queries, queryset_size, segment_length = query.shape[0], query.shape[1], query.shape[2]
        num_reward_model_inputs = num_queries * queryset_size * segment_length
        return num_reward_model_inputs

    @staticmethod
    def compute_choice_probability(total_rewards):
        return F.softmax(total_rewards)[:, 0].type(torch.float64)
