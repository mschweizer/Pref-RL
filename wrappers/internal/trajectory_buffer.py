from collections import deque

from gym import Wrapper

from wrappers.internal.experience import Experience


class TrajectoryBuffer(Wrapper):
    def __init__(self, env, trajectory_buffer_size=128):
        super().__init__(env)
        self.trajectory_buffer = deque(maxlen=trajectory_buffer_size)
        self._last_observation = None
        self._last_done = False

    def reset(self, **kwargs):
        self._last_observation = super().reset(**kwargs)
        self._last_done = False
        return self._last_observation

    def step(self, action):
        new_observation, reward, new_done, info = super().step(action)

        experience = Experience(self._last_observation, action, reward, self._last_done, info)

        self.trajectory_buffer.append(experience)

        self._last_observation = new_observation
        self._last_done = new_done

        return new_observation, reward, new_done, info
