import logging
from abc import ABC, abstractmethod

import numpy as np


class AbstractSegmentSampler(ABC):
    def __init__(self, segment_samples, trajectory_buffer, segment_length=25):
        self.segment_samples = segment_samples
        self.trajectory_buffer = trajectory_buffer
        self.segment_length = segment_length

    def try_to_sample(self):
        try:
            return self.draw_segment_sample()
        except AssertionError as e:
            logging.warning("Trajectory segment sampling failed. " + str(e))

    @abstractmethod
    def draw_segment_sample(self):
        pass


class RandomSegmentSampler(AbstractSegmentSampler):

    def draw_segment_sample(self):
        start_idx = self._get_random_start_index()
        assert len(self.trajectory_buffer) >= self.segment_length, \
            "Fewer elements in buffer than sample size."
        return list(self.trajectory_buffer)[start_idx: start_idx + self.segment_length]

    def _get_random_start_index(self):
        low = 0
        high = max(len(self.trajectory_buffer) - self.segment_length + 1, low + 1)
        return np.random.randint(low=low, high=high)
