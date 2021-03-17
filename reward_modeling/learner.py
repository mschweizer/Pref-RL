from preference_data.collection.collector import RewardMaximizingCollector
from preference_data.collection.query_selector import RandomQuerySelector
from preference_data.generation.generator import Generator
from reward_modeling.trainer import Trainer


class Learner:
    def __init__(self, policy_model, reward_model, segment_length=25, dataset_capacity=3000):
        self.query_generator = Generator(policy_model=policy_model,
                                         segment_length=segment_length)  # Problem generator
        self.query_selector = RandomQuerySelector()
        self.preference_collector = RewardMaximizingCollector(self.query_generator)  # Performance element
        self.trainer = Trainer(reward_model)  # Learning element

    def learn(self, generation_volume, with_training=True):
        self.fill_dataset(generation_volume, with_training)
        self.trainer.train(self.preference_collector.preferences)

    def fill_dataset(self, generation_volume, with_training=True):
        queries = self.query_generator.generate(generation_volume=generation_volume,
                                                with_training=with_training)
        self.preference_collector.collect_preferences(queries)
