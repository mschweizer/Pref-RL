import numpy as np
from gym import Wrapper


class VisualFeedbackRemover(Wrapper):
    def __init__(self, env):
        super(VisualFeedbackRemover, self).__init__(env)
        self.black_box = self._generate_black_box(env)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        observation = self._redact_score_area(observation)
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = self._redact_score_area(observation)
        return observation, reward, done, info

    def _generate_black_box(self, env):
        env_id = env.spec.id
        box = np.ones((84, 84, 1), dtype=np.int8)
        if "Breakout-v0" in env_id:
            # Blackbox for "Breakout-v0"
            box = self._fill_with_zeros(box, (2, 18), (6, 74))
        elif "Qbert-v0" in env_id:
            # Blackbox for "Qbert-v0"
            box = self._fill_with_zeros(box, (2, 17), (6, 40))
            box = self._fill_with_zeros(box, (6, 15), (12, 28))
        elif "BeamRider-v0" in env_id:
            # Blackbox for "BeamRider-v0"
            box = self._fill_with_zeros(box, (2, 8), (16, 78))
        return box

    @staticmethod
    def _fill_with_zeros(box, top_left, bottom_right):
        row_start, col_start = top_left
        row_end, col_end = bottom_right
        for i in range(row_start, row_end):
            for j in range(col_start, col_end):
                box[i][j][0] = 0
        return box

    def _redact_score_area(self, observation):
        return np.multiply(self.black_box, observation)

