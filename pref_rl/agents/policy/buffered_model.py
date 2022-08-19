import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from .model import PolicyModel
from ...environment_wrappers.internal.trajectory_observation.observer import TrajectoryObserver, FrameTrajectoryObserver


class VecBuffer:
    def __init__(self, vec_env: VecEnv):
        self.vec_env = vec_env
        self._buffers = self.vec_env.get_attr("trajectory_buffer")
        self.n_buffers = len(self._buffers)

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

    def update(self):
        self._buffers = self.vec_env.get_attr("trajectory_buffer")


class ObservedPolicyModel(PolicyModel):
    def __init__(self, env, train_freq, load_file=None, n_envs=1, trajectory_buffer_size=1024, human_obs=False):
        if human_obs:
            observed_env = FrameTrajectoryObserver(env, trajectory_buffer_size)
        else:
            observed_env = TrajectoryObserver(env, trajectory_buffer_size)
        super().__init__(observed_env, train_freq, load_file, n_envs)
        self.trajectory_buffer = VecBuffer(self.env)
