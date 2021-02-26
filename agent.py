from stable_baselines3 import A2C

from data_generation.preference_data_generator import PreferenceDataGenerator
from reward_modeling.preference_dataset import PreferenceDataset
from reward_modeling.reward_model import RewardModel
from reward_modeling.reward_trainer import RewardTrainer
from reward_modeling.reward_wrapper import RewardWrapper


class LearningAgent:
    def __init__(self, env, segment_length=10, simulation_steps_per_policy_update=2048, trajectory_buffer_size=10,
                 model_parameters=None):
        self.reward_model = RewardModel(env)
        if model_parameters:
            self.reward_model.load_state_dict(model_parameters)
        self.env = RewardWrapper(env=env, reward_model=self.reward_model, trajectory_buffer_size=trajectory_buffer_size)
        self.policy_model = A2C('MlpPolicy', env=self.env, n_steps=simulation_steps_per_policy_update)
        self.preference_data_generator = PreferenceDataGenerator(policy_model=self.policy_model,
                                                                 segment_length=segment_length)
        self.reward_learner = RewardTrainer(reward_model=self.reward_model)

    def choose_action(self, state):
        return self.policy_model.predict(state)

    def learn_policy(self, total_timesteps):
        self.policy_model.learn(total_timesteps)

    def learn_reward_model(self, sampling_interval=30, query_interval=50):
        preferences = self.preference_data_generator.generate(generation_volume=1000,
                                                              sampling_interval=sampling_interval,
                                                              query_interval=query_interval)
        preference_dataset = PreferenceDataset(preferences=preferences)
        self.reward_learner.train(preference_dataset)
