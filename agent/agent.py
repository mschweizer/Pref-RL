from stable_baselines3 import A2C

from reward_modeling.learner import Learner
from reward_modeling.models.reward import Reward
from wrappers.utils import add_internal_env_wrappers


class Agent:
    def __init__(self, env, segment_length=25, segment_sampling_buffer_size=100, segment_sampling_interval=30,
                 reward_standardization_buffer_size=3000, reward_standardization_std=.05, env_steps_per_rl_update=100,
                 query_generation_interval=50, reward_model_learning_rate=1e-3, reward_training_batch_size=16,
                 reward_learning_summary_writing_interval=100):
        self.reward_model = Reward(env)
        wrapped_env = add_internal_env_wrappers(env=env,
                                                reward_model=self.reward_model,
                                                segment_sampling_buffer_size=segment_sampling_buffer_size,
                                                reward_standardization_std=reward_standardization_std,
                                                reward_standardization_buffer_size=reward_standardization_buffer_size,
                                                reward_standardization_update_interval=30000)
        # TODO: Wrap stable baselines policy model in own policy model class and move env wrapping inside
        self.policy_model = A2C('MlpPolicy', env=wrapped_env, n_steps=env_steps_per_rl_update)
        self.reward_learner = Learner(policy_model=self.policy_model, reward_model=self.reward_model,
                                      segment_length=segment_length,
                                      segment_sampling_interval=segment_sampling_interval,
                                      query_generation_interval=query_generation_interval,
                                      learning_rate=reward_model_learning_rate,
                                      batch_size=reward_training_batch_size,
                                      summary_writing_interval=reward_learning_summary_writing_interval)

    def choose_action(self, state):
        return self.policy_model.predict(state)

    def learn_policy(self, total_timesteps):
        self.policy_model.learn(total_timesteps)

    def learn_reward_model(self, num_pretraining_data=0, pretraining_epochs=1):
        if self._pretraining_is_configured(num_pretraining_data, pretraining_epochs):
            self._pretrain_reward_model(num_pretraining_data, pretraining_epochs)

    def _pretrain_reward_model(self, num_pretraining_data, pretraining_epochs):
        self.reward_learner.learn(generation_volume=num_pretraining_data, epochs=pretraining_epochs,
                                  with_training=False)

    @staticmethod
    def _pretraining_is_configured(num_pretraining_data, num_pretraining_epochs):
        return num_pretraining_data > 0 and num_pretraining_epochs > 0
