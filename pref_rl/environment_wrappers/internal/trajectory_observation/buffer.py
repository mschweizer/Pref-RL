from collections import deque

from pref_rl.environment_wrappers.internal.trajectory_observation.segment import Segment, FrameSegment


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

    def append_frame_step(self, observation, action, reward, done, info, frame):
        super().append_step(observation, action, reward, done, info)
        self.frames.append(frame)

    def get_segment(self, start, stop):
        return FrameSegment(list(self.observations)[start: stop],
                            list(self.actions)[start: stop],
                            list(self.rewards)[start: stop],
                            list(self.dones)[start: stop],
                            list(self.infos)[start: stop],
                            list(self.frames)[start: stop])