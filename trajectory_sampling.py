import numpy as np


class TrajectorySegmentSampler:
    def __init__(self, trajectory_buffer, segment_length):
        self.trajectory_buffer = trajectory_buffer
        self.segment_length = segment_length

    def get_sampled_trajectory(self):
        start_idx = self.get_start_index()
        # TODO: treat case where buffer has less elements than trajectory length
        return self.trajectory_buffer.experiences[start_idx: start_idx + self.segment_length]

    def get_start_index(self):
        low = 0
        high = max(len(self.trajectory_buffer.experiences) - self.segment_length + 1, low + 1)
        return np.random.randint(low=low, high=high)
