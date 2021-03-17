from stable_baselines3 import A2C

from environment.utils import add_internal_env_wrappers
from reward_modeling.reward_learner import RewardLearner
from reward_modeling.reward_model import RewardModel


class LearningAgent:
    def __init__(self, env, segment_length=25):
        self.reward_model = RewardModel(env)
        wrapped_env = add_internal_env_wrappers(env=env, reward_model=self.reward_model,
                                                trajectory_buffer_size=100,
                                                desired_std=.05, standardization_buffer_size=3000,
                                                standardization_params_update_interval=30000)
        self.policy_model = A2C('MlpPolicy', env=wrapped_env, n_steps=100)
        self.reward_learner = RewardLearner(self.policy_model, self.reward_model, segment_length)

    def choose_action(self, state):
        return self.policy_model.predict(state)

    def learn_policy(self, total_timesteps):
        self.policy_model.learn(total_timesteps)

    def learn_reward_model(self, num_pretraining_data=0, num_pretraining_epochs=0):
        if self._pretraining_is_configured(num_pretraining_data, num_pretraining_epochs):
            self._pretrain_reward_model(num_pretraining_data, num_pretraining_epochs)

    def _pretrain_reward_model(self, num_pretraining_data, num_pretraining_epochs):
        self.reward_learner.learn(500)
        self.reward_model_trainer.train(num_epochs=num_pretraining_epochs)

    @staticmethod
    def _pretraining_is_configured(num_pretraining_data, num_pretraining_epochs):
        return num_pretraining_data > 0 and num_pretraining_epochs > 0