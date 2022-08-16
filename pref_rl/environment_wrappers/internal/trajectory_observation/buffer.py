from collections import deque

from .segment import Segment, FrameSegment


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
