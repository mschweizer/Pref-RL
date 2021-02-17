import numpy as np


class TrajectorySegmentSampler:
    def __init__(self, trajectory_buffer, segment_length):
        self.trajectory_buffer = trajectory_buffer
        self.segment_length = segment_length
        self.segment_samples = []

    def save_sample(self):
        sample = self.generate_sample()
        self.segment_samples.append(sample)

    def generate_sample(self):
        start_idx = self._get_start_index()
        assert len(self.trajectory_buffer.experiences) >= self.segment_length, \
            "Fewer elements in buffer than sample size."
        return self.trajectory_buffer.experiences[start_idx: start_idx + self.segment_length]

    def _get_start_index(self):
        low = 0
        high = max(len(self.trajectory_buffer.experiences) - self.segment_length + 1, low + 1)
        return np.random.randint(low=low, high=high)
