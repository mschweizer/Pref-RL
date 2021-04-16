import sys
from abc import ABC, abstractmethod

from preference_data.query_generation.query_generator import AbstractQueryGenerator
from preference_data.query_generation.segment.segment_sampler import AbstractSegmentSampler, RandomSegmentSampler
from preference_data.query_generation.segment.segment_sampling_callback import SegmentSamplingCallback
from preference_data.query_generation.segment.segment_selector import AbstractSegmentSelector, RandomSegmentSelector
from preference_data.query_generation.segment.utils import is_sampling_step, generation_volume_is_reached


class AbstractSegmentQueryGenerator(AbstractQueryGenerator, AbstractSegmentSampler, AbstractSegmentSelector, ABC):
    def __init__(self, policy_model, segments_per_query=2, segment_sampling_interval=30):
        AbstractQueryGenerator.__init__(self)
        # TODO: Wrap policy model in a wrapper and make trajectory buffer a property of the wrapper class
        AbstractSegmentSampler.__init__(self, trajectory_buffer=policy_model.env.envs[0].trajectory_buffer)

        self.segments_per_query = segments_per_query
        self.segment_sampling_interval = segment_sampling_interval
        self.policy_model = policy_model

        self.segment_samples = []

    def generate_queries(self, num_queries=1, with_training=True):
        num_samples = self.calculate_num_segment_samples(num_queries)
        self._generate_samples_with_training(num_samples) if with_training \
            else self._generate_samples_without_training(num_samples)
        return [self.generate_query() for _ in range(num_queries)]

    def generate_query(self):
        return self.select_segments(self.segment_samples, self.segments_per_query)

    @abstractmethod
    def calculate_num_segment_samples(self, num_queries):
        pass

    def _generate_samples_with_training(self, num_samples):
        sampling_callback = SegmentSamplingCallback(self, self.segment_sampling_interval, num_samples)
        self.policy_model.learn(total_timesteps=sys.maxsize, callback=sampling_callback)

    def _generate_samples_without_training(self, num_samples):
        num_timesteps = 0

        obs = self.policy_model.env.reset()
        while True:
            action, _states = self.policy_model.predict(obs)
            _, _, done, _ = self.policy_model.env.step(action)
            if done:
                assert False, "Env should never return Done=True because of the wrapper that should prevent this."
            if is_sampling_step(num_timesteps, self.segment_sampling_interval):
                sample = self.try_to_sample()
                if sample:
                    self.segment_samples.append(sample)
            if generation_volume_is_reached(num_samples, self.segment_samples):
                break
            num_timesteps += 1


class RandomSegmentQueryGenerator(AbstractSegmentQueryGenerator, RandomSegmentSampler, RandomSegmentSelector):
    def __init__(self, policy_model):
        super(AbstractSegmentQueryGenerator, self).__init__(policy_model)

    def calculate_num_segment_samples(self, num_queries):
        return num_queries * self.segments_per_query
