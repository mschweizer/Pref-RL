from typing import List

import numpy as np

from ..generator import AbstractAlternativeGenerator
from .....agents.policy.buffered_model import ObservedPolicyModel, VecBuffer
from .....environment_wrappers.internal.trajectory_observation.segment import Segment
from .....utils.logging import get_or_create_logger

NUM_SEGMENTS_REQUESTED_MSG = "{} segment samples requested"

START_SAMPLING_MSG = "Collecting rollout of length {} from current policy for segment sampling"

PROGRESS_MSG = "{sampled} segments sampled [{outstanding_segments} segments / {outstanding_steps} policy steps left]"


class SegmentSampler(AbstractAlternativeGenerator):
    def __init__(self, segment_length: int):
        """
        This alternative generator samples short trajectory segments from a buffer of recent agent behavior.
        :param segment_length: The length each sampled trajectory segment.
        """
        self.segment_length = segment_length
        self.logger = get_or_create_logger("SegmentSampler")

    def generate(self, policy_model: ObservedPolicyModel, num_alternatives: int) -> List[Segment]:
        """
        :param policy_model: The policy model that is used to generate the alternatives.
        :param num_alternatives: The number of alternatives that are generated.
        :return: The list of generated alternatives.
        """
        self.logger.info(NUM_SEGMENTS_REQUESTED_MSG.format(num_alternatives))

        policy_model.trajectory_buffer.update()

        outstanding_rollout_steps = \
            self._calculate_necessary_rollout_steps(num_alternatives, policy_model.trajectory_buffer)
        outstanding_segment_samples = num_alternatives
        segments_per_rollout = self._calculate_num_segments_per_rollout(policy_model.trajectory_buffer)

        segment_samples = []
        self.logger.info(START_SAMPLING_MSG.format(outstanding_rollout_steps))
        while outstanding_segment_samples > 0:
            rollout_steps_in_this_iter = \
                self._compute_num_rollout_steps_for_iteration(outstanding_rollout_steps, policy_model.trajectory_buffer)
            segment_samples_in_this_iter = \
                self._compute_num_segments_for_iteration(outstanding_segment_samples, segments_per_rollout)

            policy_model.run(rollout_steps_in_this_iter)
            policy_model.trajectory_buffer.update()
            outstanding_rollout_steps -= rollout_steps_in_this_iter

            segment_samples.extend(self._sample_segments(segment_samples_in_this_iter, policy_model.trajectory_buffer))
            outstanding_segment_samples -= segment_samples_in_this_iter

            self.logger.info(PROGRESS_MSG.format(sampled=len(segment_samples),
                                                 outstanding_segments=outstanding_segment_samples,
                                                 outstanding_steps=outstanding_rollout_steps))

        return segment_samples

    @staticmethod
    def _compute_num_segments_for_iteration(outstanding_segment_samples: int, segments_per_rollout: int) -> int:
        return min(segments_per_rollout, outstanding_segment_samples)

    def _compute_num_rollout_steps_for_iteration(self, outstanding_rollout_steps: int, buffer: VecBuffer) -> int:
        return min(buffer.size, max(int(outstanding_rollout_steps / buffer.n_buffers), self.segment_length))

    def _calculate_necessary_rollout_steps(self, num_items: int, buffer: VecBuffer) -> int:
        total_necessary_steps = 3 * num_items * self.segment_length
        return max(0, total_necessary_steps - len(buffer) * buffer.n_buffers)

    def _calculate_num_segments_per_rollout(self, buffer: VecBuffer) -> int:
        # Determine number of samples so that the probability of duplicate segment samples is fairly low
        # Source: https://github.com/nottombrown/rl-teacher/blob/b2c2201e9d2457b13185424a19da7209364f23df/rl_teacher/
        # segment_sampling.py#L76
        return max(1, int(0.3 * (len(buffer) * buffer.n_buffers) / self.segment_length))

    def _sample_segments(self, num_segments: int, buffer: VecBuffer) -> List[Segment]:
        return [self._sample_segment(buffer) for _ in range(num_segments)]

    def _sample_segment(self, trajectory_buffer: VecBuffer) -> Segment:
        start_idx = self._get_random_start_index(0, len(trajectory_buffer))
        return trajectory_buffer.get_segment(start=start_idx, end=start_idx + self.segment_length)

    def _get_random_start_index(self, start: int, end: int) -> int:
        low = start
        high = max(end - self.segment_length + 1, low + 1)
        return np.random.randint(low=low, high=high)
