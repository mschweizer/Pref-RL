import torch
from gym import Wrapper

from data_generation.experience import Experience, PredictionBuffer
from reward_modeling.reward_predictor import RewardPredictor


class RewardWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.trajectory_buffer = PredictionBuffer(size=100, prediction_stack_depth=4)  # TODO: add params to init
        self.reward_predictor = \
            RewardPredictor(env=env, num_stacked_frames=4, trajectory_buffer=self.trajectory_buffer)  # TODO: add params
        self._last_observation = None
        self._last_done = False

    def reset(self, **kwargs):
        self._last_observation = super().reset(**kwargs)
        self._last_done = False
        return self._last_observation

    def step(self, action):
        new_observation, reward, new_done, info = super().step(action)

        # A reward tensor is explicitly created because stable baselines performs a deep copy on 'info'
        # Torch otherwise throws a 'RuntimeError: Only Tensors created explicitly by the user (graph leaves)
        # support the deepcopy protocol at the moment'
        info['original_reward'] = torch.tensor(reward)

        transformed_reward = self.reward()

        # TODO: should this really be the last observation / done? see implementation of stable baselines
        experience = Experience(self._last_observation, action, transformed_reward, self._last_done, info)
        self.trajectory_buffer.append(experience)

        self._last_observation = new_observation
        self._last_done = new_done

        return new_observation, transformed_reward, new_done, info

    def reward(self):
        return self.reward_predictor.predict_utility()
