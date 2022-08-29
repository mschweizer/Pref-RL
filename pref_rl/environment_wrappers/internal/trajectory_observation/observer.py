from gym import Wrapper

from pref_rl.query_generation.choice_set_query.alternative_generation.segment_alternative.buffer import Buffer


class TrajectoryObserver(Wrapper):
    def __init__(self, env, trajectory_buffer_size=1024):
        super().__init__(env)
        self.trajectory_buffer = Buffer(buffer_size=trajectory_buffer_size)
        self._last_observation = None
        self._last_done = False

    def reset(self, **kwargs):
        self._last_observation = super().reset(**kwargs)
        self._last_done = False
        return self._last_observation

    def step(self, action):
        new_observation, reward, new_done, info = super().step(action)

        self.trajectory_buffer.append_step(self._last_observation, action, reward, self._last_done, info)

        self._last_observation = new_observation
        self._last_done = new_done

        return new_observation, reward, new_done, info
