from abc import ABC, abstractmethod

from pref_rl.query_generation.choice_set_generation.choice_item_generation.choice_item_generator import \
    AbstractQueryItemGenerator
from pref_rl.utils.logging import create_logger
from .common import RandomSamplingMixin, RandomNoResetSamplingMixin

TOO_FEW_ELEMENTS_IN_BUFFER_ERROR_MSG = "Cannot draw a segment sample of length {len} " \
                      "from a buffer with {num_elems} elements."

NEAR_DUPLICATE_SEGMENTS_WARNING_MSG = "About to sample {num_items} segments of length {seg_len} " \
           "from a trajectory of length {traj_len}. " \
           "Be aware of potential (near) duplicate segment samples."


class AbstractSegmentSampler(AbstractQueryItemGenerator, ABC):
    def __init__(self, segment_length):
        self.segment_length = segment_length
        self.logger = create_logger("SegmentSampler")

    def generate(self, policy_model, num_items):
        try:
            assert len(policy_model.trajectory_buffer) >= self.segment_length, \
                TOO_FEW_ELEMENTS_IN_BUFFER_ERROR_MSG.format(len=self.segment_length,
                                                            num_elems=len(policy_model.trajectory_buffer))
            self._log_duplicate_warning(num_items, policy_model)
            samples = []
            while len(samples) < num_items:
                samples.append(self._sample_segment(policy_model.trajectory_buffer, self.segment_length))
            return samples
        except AssertionError as e:
            self.logger.warning(
                "Trajectory segment sampling failed. " + str(e) + " Returning empty set of samples.")
            return []

    def _log_duplicate_warning(self, num_items, policy_model):
        trajectory_buffer_length = policy_model.trajectory_buffer.size
        if num_items > int(0.3 * trajectory_buffer_length / self.segment_length):
            msg = NEAR_DUPLICATE_SEGMENTS_WARNING_MSG.format(num_items=num_items,
                                                             seg_len=self.segment_length,
                                                             traj_len=trajectory_buffer_length)
            self.logger.warning(msg)

    @abstractmethod
    def _sample_segment(self, trajectory_buffer, segment_length):
        pass


class RandomSegmentSampler(RandomSamplingMixin, AbstractSegmentSampler):
    def __init__(self, segment_length):
        super().__init__(segment_length)
        self.logger = create_logger('RandomSegmentSampler')


class RandomNoResetSegmentSampler(RandomNoResetSamplingMixin, AbstractSegmentSampler):
    def __init__(self, segment_length):
        super().__init__(segment_length)
        self.logger = create_logger('RandomNoResetSegmentSampler')
