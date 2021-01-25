import numpy as np


class TrajectorySegmentSampler:
    def __init__(self, trajectory_buffer, sampling_interval, segment_length):
        self.trajectory_buffer = trajectory_buffer
        self.sampling_interval = sampling_interval
        self.trajectory_length = segment_length
        self.trajectories = []

    def sample_trajectory(self):
        sampled_trajectory = self.get_sampled_trajectory()
        self.trajectories.append(sampled_trajectory)

    def get_start_index(self):
        low = 0
        high = max(len(self.trajectory_buffer.experiences) - self.trajectory_length + 1, low + 1)
        return np.random.randint(low=low, high=high)

    def get_sampled_trajectory(self):
        start_idx = self.get_start_index()
        # TODO: treat case where sampling buffer has less elements than trajectory length
        return self.trajectory_buffer.experiences[start_idx: start_idx + self.trajectory_length]
