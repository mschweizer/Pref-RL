from abc import ABC, abstractmethod

from pref_rl.utils.logging import create_logger
from .common import RandomSamplingMixin, RandomNoResetSamplingMixin
from pref_rl.query_generation.choice_set_generation.choice_item_generation.choice_item_generator import AbstractQueryItemGenerator


class AbstractPretrainingSegmentSampler(AbstractQueryItemGenerator, ABC):
    def __init__(self, segment_length):
        self.segment_length = segment_length
        self.logger = create_logger("PretrainingSegmentSampler")

    def generate(self, policy_model, num_items):
        self.logger.info("{} segment samples requested".format(num_items))
        samples = []

        # Determine number of samples so that the probability of duplicate segment_sampling samples is fairly low
        # Source: https://github.com/nottombrown/rl-teacher/blob/b2c2201e9d2457b13185424a19da7209364f23df/rl_teacher/
        # segment_sampling.py#L76
        trajectory_buffer_length = policy_model.trajectory_buffer.size
        samples_per_rollout = max(1, int(0.3 * trajectory_buffer_length / self.segment_length))

        while len(samples) < num_items:
            self.logger.info(
                "Collecting rollout of length {} from randomly initialized policy for segment sampling".format(
                    trajectory_buffer_length))
            policy_model.run(steps=trajectory_buffer_length)
            for _ in range(samples_per_rollout):
                samples.append(self._sample_segment(policy_model.trajectory_buffer, self.segment_length))
            self.logger.info(
                "{sampled} segments sampled - {outstanding} left".format(sampled=samples_per_rollout,
                                                                         outstanding=num_items - len(samples)))
        return samples

    @abstractmethod
    def _sample_segment(self, trajectory_buffer, segment_length):
        pass


class RandomPretrainingSegmentSampler(RandomSamplingMixin, AbstractPretrainingSegmentSampler):
    pass


class RandomNoResetPretrainingSegmentSampler(RandomNoResetSamplingMixin, AbstractPretrainingSegmentSampler):
    pass
