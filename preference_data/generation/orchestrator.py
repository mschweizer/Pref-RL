from stable_baselines3.common.callbacks import EveryNTimesteps, BaseCallback


class Orchestrator:
    def __init__(self, segment_sampler, query_generator, sampling_interval=30, query_interval=50):
        self.segment_sampler = segment_sampler
        self.query_generator = query_generator
        self.sampling_interval = sampling_interval
        self.query_interval = query_interval

    def create_callbacks(self, generation_volume=3000):
        callbacks = []

        sample_trajectory = SampleTrajectoryCallback(self.segment_sampler)
        sample_callback = EveryNTimesteps(n_steps=self.sampling_interval, callback=sample_trajectory)
        callbacks.append(sample_callback)

        generate_query = GenerateQueryCallback(self.query_generator, generation_volume=generation_volume)
        # TODO: create separate "generate query interval"
        query_callback = EveryNTimesteps(n_steps=self.query_interval, callback=generate_query)
        callbacks.append(query_callback)

        return callbacks

    def is_sampling_step(self, step_num):
        return step_num % self.sampling_interval == 0

    def is_query_step(self, step_num):
        return step_num % self.query_interval == 0


class SampleTrajectoryCallback(BaseCallback):

    def __init__(self, segment_sampler, verbose=1):
        super().__init__(verbose)
        self.segment_sampler = segment_sampler

    def _on_step(self) -> bool:
        self.segment_sampler.try_save_sample()
        return True


class GenerateQueryCallback(BaseCallback):

    def __init__(self, query_generator, verbose=1, generation_volume=None):
        super().__init__(verbose)
        self.query_generator = query_generator
        self.generation_volume = generation_volume

    def _on_step(self) -> bool:
        self.query_generator.try_save_query()
        if self.generation_volume and len(self.query_generator.queries) >= self.generation_volume:
            return False
        else:
            return True
