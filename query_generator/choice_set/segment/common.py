import numpy as np


class RandomSamplingMixin:

    def _sample_segment(self, trajectory_buffer, segment_length):
        start_idx = self._get_random_start_index(len(trajectory_buffer), segment_length)
        return trajectory_buffer.get_segment(start=start_idx, stop=start_idx + segment_length)

    @staticmethod
    def _get_random_start_index(num_elements_in_buffer, segment_length):
        low = 0
        high = max(num_elements_in_buffer - segment_length + 1, low + 1)
        return np.random.randint(low=low, high=high)
