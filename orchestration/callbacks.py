import logging

from stable_baselines3.common.callbacks import BaseCallback

PREFERENCE_QUERYSET_SIZE = 2


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
        try:
            self.segment_sampler.save_sample()
        except AssertionError as e:
            logging.warning("Trajectory segment sampling failed. " + str(e))
        return True


class CollectPreferenceCallback(BaseCallback):

    def __init__(self, preference_collector, k=None, verbose=1):
        super().__init__(verbose)
        self.preference_collector = preference_collector
        self.k = k

    def _on_step(self) -> bool:
        try:
            self.preference_collector.save_preference()
        except IndexError as e:
            logging.warning("Preference collection failed. There are currently no preference queries available. "
                            "Original error message: " + str(e))
        if self.k and len(self.preference_collector.preferences) >= self.k:
            return False
        else:
            return True


class GenerateQueryCallback(BaseCallback):

    def __init__(self, query_generator, verbose=1):
        super().__init__(verbose)
        self.query_generator = query_generator

    def _on_step(self) -> bool:
        try:
            self.query_generator.save_query()
        except ValueError as e:
            log_msg = "Query generation failed. There are currently not enough ({}) segment " \
                      "samples available to create a preference query of size {}. Original error message: " + str(e)
            logging.warning(
                log_msg.format(len(self.query_generator.segment_samples), PREFERENCE_QUERYSET_SIZE))
        return True
