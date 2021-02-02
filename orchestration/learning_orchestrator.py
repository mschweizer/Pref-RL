from stable_baselines3.common.callbacks import EveryNTimesteps

from data_generation.preference_data_generator import PreferenceDataGenerator
from orchestration.callbacks import TrainRewardModelCallback, SampleTrajectoryCallback, CollectPreferenceCallback, \
    GenerateQueryCallback
from reward_modeling.reward_learner import RewardLearner


class LearningOrchestrator:
    def __init__(self, reward_model, trajectory_buffer,
                 sampling_interval=30, query_interval=50, training_interval=1000):
        self.preference_data_generator = PreferenceDataGenerator(trajectory_buffer=trajectory_buffer)
        self.reward_learner = RewardLearner(reward_model=reward_model)
        self.sampling_interval = sampling_interval
        self.query_interval = query_interval
        self.training_interval = training_interval

    def create_callbacks(self):
        callbacks = []

        train_reward_model = TrainRewardModelCallback(reward_learner=self.reward_learner)
        train_callback = EveryNTimesteps(n_steps=self.training_interval, callback=train_reward_model)
        callbacks.append(train_callback)

        sample_trajectory = SampleTrajectoryCallback(preference_data_generator=self.preference_data_generator)
        sample_callback = EveryNTimesteps(n_steps=self.sampling_interval, callback=sample_trajectory)
        callbacks.append(sample_callback)

        generate_query = GenerateQueryCallback(preference_data_generator=self.preference_data_generator)
        query_callback = EveryNTimesteps(n_steps=self.query_interval, callback=generate_query)
        # TODO: create separate "generate query interval"
        callbacks.append(query_callback)

        collect_preference = CollectPreferenceCallback(preference_data_generator=self.preference_data_generator)
        collection_callback = EveryNTimesteps(n_steps=self.query_interval, callback=collect_preference)
        callbacks.append(collection_callback)

        return callbacks
