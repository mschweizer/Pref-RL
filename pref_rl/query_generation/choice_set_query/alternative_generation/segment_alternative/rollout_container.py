from typing import Dict

from .trajectory_segment import TrajectorySegment, FrameTrajectorySegment


class RolloutContainer:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.infos = []

    def append_step(self, observation, action, reward, done, info):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.infos.append(info)

    def get_segment(self, start, end):
        return TrajectorySegment(self.observations[start: end],
                       self.actions[start: end],
                       self.rewards[start: end],
                       self.dones[start: end],
                       self.infos[start: end])

    def __len__(self):
        return len(self.observations)


class FrameRolloutContainer(RolloutContainer):
    def __init__(self):
        super().__init__()
        self.frames = []

    def append_step(self, observation, action, reward, done, info: Dict):
        assert "frame" in info
        self.frames.append(info["frame"])
        del info["frame"]
        super().append_step(observation, action, reward, done, info)

    def get_segment(self, start, end):
        return FrameTrajectorySegment(self.observations[start: end],
                            self.actions[start: end],
                            self.rewards[start: end],
                            self.dones[start: end],
                            self.infos[start: end],
                            self.frames[start: end])
