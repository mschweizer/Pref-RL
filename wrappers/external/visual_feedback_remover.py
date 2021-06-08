from gym import Wrapper
import numpy as np


class VisualFeedbackRemover(Wrapper):
    def __init__(self, env):
        super(VisualFeedbackRemover, self).__init__(env)
        self.black_matrix = np.ones((84, 84, 1), dtype=np.int8)
        for i in range(2, 6):
            for j in range(18, 74):
                self.black_matrix[i][j][0] = 0

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        observation = np.multiply(self.black_matrix, observation)
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = np.multiply(self.black_matrix, observation)
        return observation, reward, done, info
