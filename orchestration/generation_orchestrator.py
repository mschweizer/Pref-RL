from stable_baselines3.common.callbacks import EveryNTimesteps

from orchestration.callbacks import SampleTrajectoryCallback, GenerateQueryCallback, CollectPreferenceCallback


class GenerationOrchestrator:
    def __init__(self, segment_sampler, query_generator, preference_collector):
        self.segment_sampler = segment_sampler
        self.query_generator = query_generator
        self.preference_collector = preference_collector

    def create_callbacks(self, generation_volume=None, sampling_interval=30, query_interval=50):
        callbacks = []

        sample_trajectory = SampleTrajectoryCallback(self.segment_sampler)
        sample_callback = EveryNTimesteps(n_steps=sampling_interval, callback=sample_trajectory)
        callbacks.append(sample_callback)

        generate_query = GenerateQueryCallback(self.query_generator)
        query_callback = EveryNTimesteps(n_steps=query_interval, callback=generate_query)
        # TODO: create separate "generate query interval"
        callbacks.append(query_callback)

        collect_preference = CollectPreferenceCallback(self.preference_collector, generation_volume)
        collection_callback = EveryNTimesteps(n_steps=query_interval, callback=collect_preference)
        callbacks.append(collection_callback)

        return callbacks
