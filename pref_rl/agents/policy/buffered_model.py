from typing import List

import numpy as np

from .model import PolicyModel
from ...environment_wrappers.internal.trajectory_observation.buffer import Buffer
from ...environment_wrappers.internal.trajectory_observation.observer import TrajectoryObserver, FrameTrajectoryObserver


class VecBuffer:
    def __init__(self, atomic_buffers: List[Buffer]):
        self.buffers = atomic_buffers

    def get_segment(self, start: int, end: int, env_idx: int = None):
        if not env_idx:
            env_idx = np.random.randint(self.n_buffers)
        return self.buffers[env_idx].get_segment(start, end)

    def __len__(self):
        return len(self.buffers[0])

    @property
    def size(self):
        return self.buffers[0].size

    @property
    def n_buffers(self):
        return len(self.buffers)


class ObservedPolicyModel(PolicyModel):
    def __init__(self, env, train_freq, load_file=None, n_envs=1, trajectory_buffer_size=1024, human_obs=False):
        if human_obs:
            observed_env = FrameTrajectoryObserver(env, trajectory_buffer_size)
        else:
            observed_env = TrajectoryObserver(env, trajectory_buffer_size)
        super().__init__(observed_env, train_freq, load_file, n_envs)
        self.trajectory_buffer = VecBuffer(self.env.get_attr("trajectory_buffer"))
