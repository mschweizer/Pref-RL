import numpy as np


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
