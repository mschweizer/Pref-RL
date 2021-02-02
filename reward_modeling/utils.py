import numpy as np
import torch


class Preprocessor:
    def __init__(self, env, num_stacked_frames):
        self.env = env
        self.num_stacked_frames = num_stacked_frames

    def prepare_data(self, prediction_buffer):
        data = self.create_empty_data_array()
        for i, experience in enumerate(reversed(prediction_buffer)):
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
        if len(self.env.observation_space.shape) > 1:
            observation = observation.ravel()
        return observation

    def convert_action_to_array(self, action):
        if len(self.env.action_space.shape) > 1:
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
        return int(np.prod(self.env.observation_space.shape))

    def get_flattened_action_space_length(self):
        return int(np.prod(self.env.action_space.shape))
