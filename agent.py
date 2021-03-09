from stable_baselines3 import A2C

from reward_modeling.reward_model import RewardModel
from reward_modeling.reward_model_trainer import RewardModelTrainer
from reward_modeling.reward_wrapper import RewardWrapper


class LearningAgent:
    def __init__(self, env, segment_length=25, simulation_steps_per_policy_update=2048, trajectory_buffer_size=100,
                 model_parameters=None):
        self.reward_model = RewardModel(env)
        if model_parameters:
            self.reward_model.load_state_dict(model_parameters)
        self.env = RewardWrapper(env=env, reward_model=self.reward_model, trajectory_buffer_size=trajectory_buffer_size)
        self.policy_model = A2C('MlpPolicy', env=self.env, n_steps=simulation_steps_per_policy_update)
        self.reward_model_trainer = RewardModelTrainer(policy_model=self.policy_model, reward_model=self.reward_model,
                                                       segment_length=segment_length)

    def choose_action(self, state):
        return self.policy_model.predict(state)

    def learn_policy(self, total_timesteps):
        self.policy_model.learn(total_timesteps)

    def learn_reward_model(self, sampling_interval=30, query_interval=50, num_pretraining_data=0,
                           num_pretraining_epochs=0):
        if self._pretraining_is_configured(num_pretraining_data, num_pretraining_epochs):
            self._pretrain_reward_model(num_pretraining_data, num_pretraining_epochs, sampling_interval, query_interval)

    def _pretrain_reward_model(self, num_pretraining_data, num_pretraining_epochs, sampling_interval, query_interval):
        self.reward_model_trainer.fill_dataset(generation_volume=num_pretraining_data,
                                               sampling_interval=sampling_interval,
                                               query_interval=query_interval,
                                               with_training=False)
        self.reward_model_trainer.train(num_epochs=num_pretraining_epochs)

    @staticmethod
    def _pretraining_is_configured(num_pretraining_data, num_pretraining_epochs):
        return num_pretraining_data > 0 and num_pretraining_epochs > 0
