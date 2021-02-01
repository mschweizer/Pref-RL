import logging

from stable_baselines3.common.callbacks import BaseCallback

PREFERENCE_QUERYSET_SIZE = 2


class TrainRewardModelCallback(BaseCallback):

    def __init__(self, reward_learner, verbose=1):
        super().__init__(verbose)
        self.reward_learner = reward_learner

    def _on_step(self) -> bool:
        self.reward_learner.learn()
        return True


class SampleTrajectoryCallback(BaseCallback):

    def __init__(self, preference_data_generator, verbose=1):
        super().__init__(verbose)
        self.preference_data_generator = preference_data_generator

    def _on_step(self) -> bool:
        try:
            self.preference_data_generator.generate_sample()
        except AssertionError as e:
            logging.warning("Trajectory segment sampling failed. " + str(e))
        return True


class CollectPreferenceCallback(BaseCallback):

    def __init__(self, preference_data_generator, verbose=1):
        super().__init__(verbose)
        self.preference_data_generator = preference_data_generator

    def _on_step(self) -> bool:
        try:
            self.preference_data_generator.collect_preference()
        except IndexError as e:
            logging.warning("Preference collection failed. There are currently no preference queries available. "
                            "Original error message: " + str(e))
        return True


class GenerateQueryCallback(BaseCallback):

    def __init__(self, preference_data_generator, verbose=1):
        super().__init__(verbose)
        self.preference_data_generator = preference_data_generator

    def _on_step(self) -> bool:
        try:
            self.preference_data_generator.generate_query()
        except ValueError as e:
            log_msg = "Query generation failed. There are currently not enough ({}) segment " \
                      "samples available to create a preference query of size {}. Original error message: " + str(e)
            logging.warning(
                log_msg.format(len(self.preference_data_generator.segment_samples), PREFERENCE_QUERYSET_SIZE))
        return True
