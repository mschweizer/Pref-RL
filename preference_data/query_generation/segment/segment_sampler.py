import logging
from abc import ABC, abstractmethod

import numpy as np


class AbstractSegmentSampler(ABC):
    def __init__(self, trajectory_buffer, segment_length=25):
        self.trajectory_buffer = trajectory_buffer
        self.segment_length = segment_length

    def try_to_sample(self):
        try:
            return self.generate_sample()
        except AssertionError as e:
            logging.warning("Trajectory segment sampling failed. " + str(e))

    @abstractmethod
    def generate_sample(self):
        pass


class RandomSegmentSampler(AbstractSegmentSampler):

    def generate_sample(self):
        start_idx = self._get_start_index()
        assert len(self.trajectory_buffer) >= self.segment_length, \
            "Fewer elements in buffer than sample size."
        return list(self.trajectory_buffer)[start_idx: start_idx + self.segment_length]

    def _get_start_index(self):
        low = 0
        high = max(len(self.trajectory_buffer) - self.segment_length + 1, low + 1)
        return np.random.randint(low=low, high=high)
