import logging
from abc import ABC, abstractmethod

import numpy as np

from query_generator.query_item_generator import AbstractQueryItemGenerator


class AbstractSegmentSampler(AbstractQueryItemGenerator, ABC):
    def __init__(self, segment_length):
        self.segment_length = segment_length

    def generate(self, policy_model, num_items):
        try:
            assert len(policy_model.trajectory_buffer) >= self.segment_length, \
                "Cannot draw a segment sample of length {len} " \
                "from a buffer with {num_elems} elements.".format(len=self.segment_length,
                                                                  num_elems=len(policy_model.trajectory_buffer))
            self._log_duplicate_warning(num_items, policy_model)
            samples = []
            while len(samples) < num_items:
                samples.append(self._sample_segment(policy_model.trajectory_buffer, self.segment_length))
            return samples
        except AssertionError as e:
            logging.warning("Trajectory segment sampling failed. " + str(e) +
                            " Returning empty set of samples.")
            return []

    def _log_duplicate_warning(self, num_items, policy_model):
        trajectory_buffer_length = policy_model.trajectory_buffer.size
        if num_items > int(0.3 * trajectory_buffer_length / self.segment_length):
            msg = "About to sample {num_items} segments of length {seg_len} " \
                  "from a trajectory of length {traj_len}. " \
                  "Be aware of potential (near) duplicate segment samples.".format(num_items=num_items,
                                                                                   seg_len=self.segment_length,
                                                                                   traj_len=trajectory_buffer_length)
            logging.warning(msg)

    @abstractmethod
    def _sample_segment(self, trajectory_buffer, segment_length):
        pass


class RandomSegmentSampler(AbstractSegmentSampler):

    def _sample_segment(self, trajectory_buffer, segment_length):
        start_idx = self._get_random_start_index(len(trajectory_buffer), segment_length)
        return trajectory_buffer.get_segment(start=start_idx, stop=start_idx + segment_length)

    @staticmethod
    def _get_random_start_index(num_elements_in_buffer, segment_length):
        low = 0
        high = max(num_elements_in_buffer - segment_length + 1, low + 1)
        return np.random.randint(low=low, high=high)
