import logging

from stable_baselines3.common.callbacks import BaseCallback

PREFERENCE_QUERYSET_SIZE = 2


class TrainRewardModelCallback(BaseCallback):

    def __init__(self, agent, verbose=1):
        super().__init__(verbose)
        self.agent = agent

    def _on_step(self) -> bool:
        self.agent.train_reward_model()
        return True


class SampleTrajectoryCallback(BaseCallback):

    def __init__(self, agent, verbose=1):
        super().__init__(verbose)
        self.agent = agent

    def _on_step(self) -> bool:
        self.agent.sample_trajectory()
        return True


class QueryPreferenceCallback(BaseCallback):

    def __init__(self, agent, verbose=1):
        super().__init__(verbose)
        self.agent = agent

    def _on_step(self) -> bool:
        if len(self.agent.segment_samples) >= PREFERENCE_QUERYSET_SIZE:
            self.agent.query_preference()
        else:
            logging.warning("There are currently {nsamples} segment samples available. "
                            "These are too few to create a preference query of size {qsize}. "
                            "Skipping query generation until enough "
                            "samples are available.".format(nsamples=len(self.agent.segment_samples),
                                                            qsize=PREFERENCE_QUERYSET_SIZE))
        return True
