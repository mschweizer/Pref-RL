import numpy as np

from .sampler import AbstractSegmentSampler
from pref_rl.utils.logging import create_logger


class RandomSegmentSampler(AbstractSegmentSampler):
    def __init__(self, segment_length):
        super().__init__(segment_length)
        self.logger = create_logger('RandomSegmentSampler')

    def _sample_segment(self, trajectory_buffer):
        start_idx = self._get_random_start_index(0, len(trajectory_buffer))
        return trajectory_buffer.get_segment(start=start_idx, stop=start_idx + self.segment_length)

    def _get_random_start_index(self, start, end):
        low = start
        high = max(end - self.segment_length + 1, low + 1)
        return np.random.randint(low=low, high=high)