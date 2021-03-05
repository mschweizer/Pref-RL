from stable_baselines3.common.callbacks import BaseCallback


class TrainRewardModelCallback(BaseCallback):

    def __init__(self, reward_learner, verbose=1):
        super().__init__(verbose)
        self.reward_learner = reward_learner

    def _on_step(self) -> bool:
        self.reward_learner.train()
        return True


class SampleTrajectoryCallback(BaseCallback):

    def __init__(self, segment_sampler, verbose=1):
        super().__init__(verbose)
        self.segment_sampler = segment_sampler

    def _on_step(self) -> bool:
        self.segment_sampler.try_save_sample()
        return True


class CollectPreferenceCallback(BaseCallback):

    def __init__(self, preference_collector, generation_volume=None, verbose=1):
        super().__init__(verbose)
        self.preference_collector = preference_collector
        self.generation_volume = generation_volume

    def _on_step(self) -> bool:
        self.preference_collector.try_save_preference()
        if self.generation_volume and len(self.preference_collector.preferences) >= self.generation_volume:
            return False
        else:
            return True


class GenerateQueryCallback(BaseCallback):

    def __init__(self, query_generator, verbose=1):
        super().__init__(verbose)
        self.query_generator = query_generator

    def _on_step(self) -> bool:
        self.query_generator.try_save_query()
        return True
