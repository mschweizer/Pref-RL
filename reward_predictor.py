import numpy as np
import torch

from reward_net import RewardNet


class RewardPredictor:
    def __init__(self, env, trajectory_buffer, num_stacked_frames, training_interval, model_parameters=None):
        self.environment = env
        self.prediction_buffer = trajectory_buffer.prediction_context
        self.num_stacked_frames = num_stacked_frames
        self.training_interval = training_interval
        self.reward_net = RewardNet(self.get_flattened_input_length())
        if model_parameters:
            self.reward_net.load_state_dict(model_parameters)

    def predict_utility(self):
        input_data = self.prepare_data()
        return self.reward_net(input_data)

    def prepare_data(self):
        data = self.create_empty_data_array()
        for i, experience in enumerate(reversed(self.prediction_buffer)):
            experience = self.convert_experience_to_array(experience)
            data = self.add_experience(data, experience, i)
        return data

    def create_empty_data_array(self):
        return torch.zeros(self.get_flattened_input_length(), dtype=torch.float32)

    def add_experience(self, data, experience_array, i):
        experience_length = self.get_flattened_experience_length()
        if i == 0:
            data[-experience_length:] = experience_array
        else:
            data[-(i + 1) * experience_length:-i * experience_length] = experience_array

        return data

    def convert_experience_to_array(self, experience):
        observation = self.convert_observation_to_array(experience.observation)
        action = self.convert_action_to_array(experience.action)
        return self.combine_arrays(observation, action)

    def convert_observation_to_array(self, observation):
        if len(self.environment.observation_space.shape) > 1:
            observation = observation.ravel()
        return observation

    def convert_action_to_array(self, action):
        if len(self.environment.action_space.shape) > 1:
            action = action.ravel()
        return np.array(action)

    @staticmethod
    def combine_arrays(observation, action):
        return torch.from_numpy(np.hstack((observation, action)))

    def get_flattened_input_length(self):
        return self.num_stacked_frames * self.get_flattened_experience_length()

    def get_flattened_experience_length(self):
        return self.get_flattened_action_space_length() + self.get_flattened_observation_space_length()

    def get_flattened_observation_space_length(self):
        return int(np.prod(self.environment.observation_space.shape))

    def get_flattened_action_space_length(self):
        return int(np.prod(self.environment.action_space.shape))
