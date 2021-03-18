from preference_data.collection.collector import RewardMaximizingCollector
from preference_data.collection.query_selector import RandomQuerySelector
from preference_data.generation.generator import Generator
from reward_modeling.trainer import Trainer


class Learner:
    def __init__(self, policy_model, reward_model, segment_length=25, dataset_capacity=3000,
                 segment_sampling_interval=30, query_generation_interval=50, learning_rate=1e-3,
                 summary_writing_interval=100, batch_size=16):
        self.query_generator = Generator(policy_model=policy_model,
                                         segment_length=segment_length,
                                         segment_sampling_interval=segment_sampling_interval,
                                         query_generation_interval=query_generation_interval)  # Problem generator
        self.query_selector = RandomQuerySelector()
        self.preference_collector = RewardMaximizingCollector(self.query_generator,
                                                              num_preferences=dataset_capacity)  # Performance element
        self.trainer = Trainer(reward_model=reward_model,
                               learning_rate=learning_rate,
                               batch_size=batch_size,
                               summary_writing_interval=summary_writing_interval)  # Learning element

    def learn(self, generation_volume, epochs=1, with_training=True):
        self.fill_dataset(generation_volume, with_training)
        self.trainer.train(self.preference_collector.preferences, epochs)

    def fill_dataset(self, generation_volume, with_training=True):
        queries = self.query_generator.generate(generation_volume=generation_volume,
                                                with_training=with_training)
        self.preference_collector.collect_preferences(queries)
