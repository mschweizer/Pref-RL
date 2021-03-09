from stable_baselines3.common.callbacks import EveryNTimesteps

from orchestration.callbacks import TrainRewardModelCallback
from reward_modeling.reward_model_trainer import RewardModelTrainer


class LearningOrchestrator:

    def __init__(self, reward_model, policy_model):
        self.reward_trainer = RewardModelTrainer(reward_model=reward_model, policy_model=policy_model)

    def create_callback(self, training_interval=1000):
        train_reward_model = TrainRewardModelCallback(reward_learner=self.reward_trainer)
        train_callback = EveryNTimesteps(n_steps=training_interval, callback=train_reward_model)

        return train_callback
