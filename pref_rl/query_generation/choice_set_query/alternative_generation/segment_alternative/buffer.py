from collections import deque
from typing import Tuple, Dict

import numpy as np
from numpy import ndarray

from pref_rl.query_generation.choice_set_query.alternative_generation.segment_alternative.segment import Segment, \
    FrameSegment


class VecBuffer:
    def __init__(self, num_envs: int, image_obs: bool = False):
        if image_obs:
            self._buffers = [FrameBuffer(buffer_size=1000) for _ in range(num_envs)]
        else:
            self._buffers = [Buffer(buffer_size=1000) for _ in range(num_envs)]
        self.n_buffers = num_envs

    def get_segment(self, start: int, end: int, env_idx: int = None):
        if not env_idx:
            env_idx = np.random.randint(self.n_buffers)
        return self._buffers[env_idx].get_segment(start, end)

    def get_buffer(self, idx: int):
        return self._buffers[idx]

    def __len__(self):
        return len(self._buffers[0])

    @property
    def size(self):
        return self._buffers[0].size

    def append_step(self, observation: ndarray, action: ndarray, reward: ndarray, done: ndarray, info: Tuple):
        for i in range(self.n_buffers):
            self._buffers[i].append_step(observation[i], action[i], reward[i], done[i], info[i])


class Buffer:
    def __init__(self, buffer_size):
        # TODO: change deque -> ndarray (also in FrameBuffer)
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

    def get_segment(self, start, end):
        return Segment(list(self.observations)[start: end],
                       list(self.actions)[start: end],
                       list(self.rewards)[start: end],
                       list(self.dones)[start: end],
                       list(self.infos)[start: end])

    def __len__(self):
        return len(self.observations)

    @property
    def size(self):
        return self.observations.maxlen


class FrameBuffer(Buffer):
    def __init__(self, buffer_size):
        super().__init__(buffer_size)
        self.frames = deque(maxlen=buffer_size)

    def append_step(self, observation, action, reward, done, info: Dict):
        assert "frame" in info
        self.frames.append(info["frame"])
        del info["frame"]
        super().append_step(observation, action, reward, done, info)

    def append_frame_step(self, observation, action, reward, done, info, frame):
        super().append_step(observation, action, reward, done, info)
        self.frames.append(frame)

    def get_segment(self, start, end):
        return FrameSegment(list(self.observations)[start: end],
                            list(self.actions)[start: end],
                            list(self.rewards)[start: end],
                            list(self.dones)[start: end],
                            list(self.infos)[start: end],
                            list(self.frames)[start: end])
