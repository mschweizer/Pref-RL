import numpy as np

from .sampler import AbstractSegmentSampler
from pref_rl.utils.logging import create_logger


class RandomSegmentSampler(AbstractSegmentSampler):
    def __init__(self, segment_length):
        super().__init__(segment_length)
        self.logger = create_logger('RandomSegmentSampler')

    def _sample_segment(self, trajectory_buffer, segment_length):
        start_idx = self._get_random_start_index(len(trajectory_buffer), segment_length)
        return trajectory_buffer.get_segment(start=start_idx, stop=start_idx + segment_length)

    @staticmethod
    def _get_random_start_index(num_elements_in_buffer, segment_length):
        low = 0
        high = max(num_elements_in_buffer - segment_length + 1, low + 1)
        return np.random.randint(low=low, high=high)