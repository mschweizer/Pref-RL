import sys

from data_generation.preference_collector import RewardMaximizingPreferenceCollector
from data_generation.query_generator import RandomQueryGenerator
from data_generation.segment_sampler import TrajectorySegmentSampler
from orchestration.generation_orchestrator import GenerationOrchestrator


class PreferenceDataGenerator:

    def __init__(self, policy_model, segment_length=25):
        self.segment_sampler = TrajectorySegmentSampler(policy_model.env.envs[0].trajectory_buffer, segment_length)
        self.query_generator = RandomQueryGenerator(self.segment_sampler.segment_samples)
        self.preference_collector = RewardMaximizingPreferenceCollector(self.query_generator.queries)

        self.policy_model = policy_model
        self.orchestrator = GenerationOrchestrator(self.segment_sampler, self.query_generator,
                                                   self.preference_collector)

    def generate(self, generation_volume, sampling_interval, query_interval):
        self.clear_generated_data()
        callbacks = self.orchestrator.create_callbacks(generation_volume, sampling_interval, query_interval)
        self.policy_model.learn(total_timesteps=sys.maxsize, callback=callbacks)
        return self.preference_collector.preferences

    def clear_generated_data(self):
        self.segment_sampler.segment_samples.clear()
        self.query_generator.queries.clear()
        self.preference_collector.preferences.clear()
