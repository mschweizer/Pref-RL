import logging

import numpy as np


class SegmentSampler:
    def __init__(self, trajectory_buffer, segment_length=25):
        self.trajectory_buffer = trajectory_buffer
        self.segment_length = segment_length
        self.segment_samples = []

    def try_save_sample(self):
        try:
            self.save_sample()
        except AssertionError as e:
            logging.warning("Trajectory segment sampling failed. " + str(e))

    def save_sample(self):
        sample = self.generate_sample()
        self.segment_samples.append(sample)

    def generate_sample(self):
        start_idx = self._get_start_index()
        assert len(self.trajectory_buffer) >= self.segment_length, \
            "Fewer elements in buffer than sample size."
        return list(self.trajectory_buffer)[start_idx: start_idx + self.segment_length]

    def clear(self):
        self.segment_samples.clear()

    def _get_start_index(self):
        low = 0
        high = max(len(self.trajectory_buffer) - self.segment_length + 1, low + 1)
        return np.random.randint(low=low, high=high)
