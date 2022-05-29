from abc import ABC, abstractmethod

from .common import RandomSamplingMixin
from ...query_item_generator import AbstractQueryItemGenerator


class AbstractPretrainingSegmentSampler(AbstractQueryItemGenerator, ABC):
    def __init__(self, segment_length):
        self.segment_length = segment_length

    def generate(self, policy_model, num_items):
        samples = []

        # Determine number of samples so that the probability of duplicate segment samples is fairly low
        # Source: https://github.com/nottombrown/rl-teacher/blob/b2c2201e9d2457b13185424a19da7209364f23df/rl_teacher/
        # segment_sampling.py#L76
        trajectory_buffer_length = policy_model.trajectory_buffer.size
        samples_per_rollout = max(1, int(0.3 * trajectory_buffer_length / self.segment_length))

        while len(samples) < num_items:
            policy_model.run(steps=policy_model.trajectory_buffer.size)
            for _ in range(samples_per_rollout):
                samples.append(self._sample_segment(policy_model.trajectory_buffer, self.segment_length))
        return samples

    @abstractmethod
    def _sample_segment(self, trajectory_buffer, segment_length):
        pass


class RandomPretrainingSegmentSampler(RandomSamplingMixin, AbstractPretrainingSegmentSampler):
    pass
