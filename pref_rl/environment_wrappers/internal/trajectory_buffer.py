from collections import deque

import numpy as np
from gym import Wrapper


class TrajectoryBuffer(Wrapper):
    def __init__(self, env, trajectory_buffer_size=1024):  # TODO: couple buffer size with steps per RL model update
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


class FrameTrajectoryBuffer(Wrapper):
    def __init__(self, env, trajectory_buffer_size=128):
        super().__init__(env)
        self.trajectory_buffer = FrameBuffer(buffer_size=trajectory_buffer_size)
        self._last_observation = None
        self._last_done = False

    def step(self, action):
        _last_image = self.env.render(mode='rgb_array')

        new_observation, reward, new_done, info = super().step(action)

        self.trajectory_buffer.append_step(
            self._last_observation, action, reward, self._last_done, info, _last_image)

        self._last_observation = new_observation
        self._last_done = new_done

        return new_observation, reward, new_done, info


class Buffer:
    def __init__(self, buffer_size):
        self.observations = deque(maxlen=buffer_size)
        self.actions = deque(maxlen=buffer_size)
        self.rewards = deque(maxlen=buffer_size)
        self.dones = deque(maxlen=buffer_size)
        self.infos = deque(maxlen=buffer_size)

    def append_step(self, observation, action, reward, done, info):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.infos.append(info)

    def get_segment(self, start, stop):
        return Segment(list(self.observations)[start: stop],
                       list(self.actions)[start: stop],
                       list(self.rewards)[start: stop],
                       list(self.dones)[start: stop],
                       list(self.infos)[start: stop])

    def __len__(self):
        return len(self.observations)

    @property
    def size(self):
        return self.observations.maxlen


class FrameBuffer(Buffer):
    def __init__(self, buffer_size):
        super().__init__(buffer_size)
        self.frames = deque(maxlen=buffer_size)

    def append_step(self, observation, action, reward, done, info, frame):
        super().append_step(observation, action, reward, done, info)
        self.frames.append(frame)

    def get_segment(self, start, stop):
        return FrameSegment(list(self.observations)[start: stop],
                            list(self.actions)[start: stop],
                            list(self.rewards)[start: stop],
                            list(self.dones)[start: stop],
                            list(self.infos)[start: stop],
                            list(self.frames)[start: stop])


class Segment:
    def __init__(self, observations, actions, rewards, dones, infos):
        self.observations = np.array(observations)
        self.actions = np.array(actions)
        self.rewards = np.array(rewards)
        self.dones = np.array(dones)
        self.infos = infos

    def __len__(self):
        return len(self.infos)

    def get_step(self, idx):
        return {"observation": self.observations[idx],
                "action": self.actions[idx],
                "reward": self.rewards[idx],
                "done": self.dones[idx],
                "info": self.infos[idx]}

                
class FrameSegment(Segment):
    def __init__(self, observations, actions, rewards, dones, infos, frames):
        super().__init__(observations, actions, rewards, dones, infos)
        self.frames = np.array(frames)

    def get_step(self, idx):
        tmpdict = super().get_step(idx)
        tmpdict['frame'] = self.frames[idx]
        return tmpdict
        