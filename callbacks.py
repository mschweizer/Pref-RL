from stable_baselines3.common.callbacks import BaseCallback


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
