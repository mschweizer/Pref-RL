from data_generation.data_generator import DataGenerator
from data_generation.preference_collector import RewardMaximizingPreferenceCollector
from data_generation.query_selector import RandomQuerySelector
from reward_modeling.reward_model_trainer import ModelTrainer


class RewardLearner:
    def __init__(self, policy_model, reward_model, segment_length=25, dataset_capacity=3000):
        self.query_generator = DataGenerator(policy_model=policy_model,
                                             segment_length=segment_length)  # Problem generator
        self.query_selector = RandomQuerySelector()
        self.preference_collector = RewardMaximizingPreferenceCollector(self.query_generator)  # Performance element
        self.trainer = ModelTrainer(reward_model)  # Learning element

    def learn(self, generation_volume, with_training=True):
        self.fill_dataset(generation_volume, with_training)
        self.trainer.train(self.preference_collector.preferences)

    def fill_dataset(self, generation_volume, with_training=True):
        queries = self.query_generator.generate(generation_volume=generation_volume,
                                                with_training=with_training)
        self.preference_collector.collect_preferences(queries)
