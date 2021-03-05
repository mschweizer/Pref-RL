import sys

from data_generation.preference_collector import RewardMaximizingPreferenceCollector
from data_generation.query_generator import RandomQueryGenerator
from data_generation.segment_sampler import TrajectorySegmentSampler
from orchestration.generation_orchestrator import GenerationOrchestrator


class PreferenceDataGenerator:

    def __init__(self, policy_model, segment_length=25):
        trajectory_buffer = policy_model.env.envs[0].trajectory_buffer
        self.segment_sampler = TrajectorySegmentSampler(trajectory_buffer, segment_length)
        self.query_generator = RandomQueryGenerator(self.segment_sampler.segment_samples)
        self.preference_collector = RewardMaximizingPreferenceCollector(self.query_generator.queries)

        self.policy_model = policy_model
        self.orchestrator = GenerationOrchestrator(self.segment_sampler, self.query_generator,
                                                   self.preference_collector)

    def generate(self, generation_volume, sampling_interval, query_interval, with_training=True):
        self.clear()
        if with_training:
            self._generate_with_training(generation_volume, sampling_interval, query_interval)
        else:
            self._generate_without_training(generation_volume, sampling_interval, query_interval)
        return self.preference_collector.preferences

    def _generate_with_training(self, generation_volume, sampling_interval, query_interval):
        callbacks = self.orchestrator.create_callbacks(generation_volume, sampling_interval, query_interval)
        self.policy_model.learn(total_timesteps=sys.maxsize, callback=callbacks)

    def _generate_without_training(self, generation_volume, sampling_interval, query_interval):
        num_timesteps = 0

        obs = self.policy_model.env.reset()
        while num_timesteps < sys.maxsize:
            action, _states = self.policy_model.predict(obs)
            _, _, done, _ = self.policy_model.env.step(action)
            if done:
                assert False, "Env should never return Done=True because of the wrapper that should prevent this."
            if num_timesteps % sampling_interval == 0:
                self.segment_sampler.try_save_sample()
            if num_timesteps % query_interval == 0:
                self.query_generator.try_save_query()
                self.preference_collector.try_save_preference()
            if generation_volume and len(self.preference_collector.preferences) >= generation_volume:
                break

    def clear(self):
        self.segment_sampler.segment_samples.clear()
        self.query_generator.queries.clear()
        self.preference_collector.preferences.clear()
