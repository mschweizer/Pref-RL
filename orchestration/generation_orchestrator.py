from stable_baselines3.common.callbacks import EveryNTimesteps

from orchestration.callbacks import SampleTrajectoryCallback, GenerateQueryCallback, CollectPreferenceCallback


class GenerationOrchestrator:
    def __init__(self, segment_sampler, query_generator, preference_collector, sampling_interval=30, query_interval=50):
        self.segment_sampler = segment_sampler
        self.query_generator = query_generator
        self.preference_collector = preference_collector
        self.sampling_interval = sampling_interval
        self.query_interval = query_interval

    def create_callbacks(self, generation_volume=3000):
        callbacks = []

        sample_trajectory = SampleTrajectoryCallback(self.segment_sampler)
        sample_callback = EveryNTimesteps(n_steps=self.sampling_interval, callback=sample_trajectory)
        callbacks.append(sample_callback)

        generate_query = GenerateQueryCallback(self.query_generator)
        # TODO: create separate "generate query interval"
        query_callback = EveryNTimesteps(n_steps=self.query_interval, callback=generate_query)
        callbacks.append(query_callback)

        collect_preference = CollectPreferenceCallback(self.preference_collector, generation_volume)
        collection_callback = EveryNTimesteps(n_steps=self.query_interval, callback=collect_preference)
        callbacks.append(collection_callback)

        return callbacks

    def is_sampling_step(self, step_num):
        return step_num % self.sampling_interval == 0

    def is_query_step(self, step_num):
        return step_num % self.query_interval == 0
