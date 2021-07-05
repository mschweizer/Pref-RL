from gym import Wrapper
import numpy as np


class VisualFeedbackRemover(Wrapper):
    def __init__(self, env):
        super(VisualFeedbackRemover, self).__init__(env)
        self.black_matrix = np.ones((84, 84, 1), dtype=np.int8)
        env_id = env.spec.id
        if "Breakout-v0" in env_id:
            # Blackbox for "Breakout-v0"
            for i in range(2, 6):
                for j in range(18, 74):
                    self.black_matrix[i][j][0] = 0
        elif "Qbert-v0" in env_id:
            # Blackbox for "Qbert-v0"
            for i in range(2, 6):
                for j in range(17, 40):
                    self.black_matrix[i][j][0] = 0
            for i in range(6, 12):
                for j in range(15, 28):
                    self.black_matrix[i][j][0] = 0
        elif "BeamRider-v0" in env_id:
            # Blackbox for "BeamRider-v0"
            for i in range(2, 16):
                for j in range(8, 78):
                    self.black_matrix[i][j][0] = 0
        print(f"Blackmatrix für {env_id} erzeugt")

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        observation = np.multiply(self.black_matrix, observation)
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = np.multiply(self.black_matrix, observation)
        return observation, reward, done, info
