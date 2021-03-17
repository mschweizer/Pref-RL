from abc import ABC, abstractmethod

from preference_data.dataset import Dataset
from preference_data.preference.label import Label


class Collector(ABC):

    def __init__(self, queries, num_preferences=3000):
        self.queries = queries
        self.preferences = Dataset(capacity=num_preferences)

    def collect_preferences(self, queries):
        self.preferences.extend([(query, self.collect_preference(query)) for query in queries])

    @abstractmethod
    def collect_preference(self, query):
        """Returns preference (enum)"""


class RandomCollector(Collector):

    def collect_preference(self, query):
        return Label.random()


class RewardMaximizingCollector(Collector):

    def collect_preference(self, query):
        reward_1, reward_2 = self.compute_total_rewards(query)
        return self.compute_preference(reward_1, reward_2)

    @staticmethod
    def compute_total_rewards(query):
        return (sum(experience.info["original_reward"] for experience in segment) for segment in query)

    @staticmethod
    def compute_preference(reward_1, reward_2):
        if reward_1 > reward_2:
            return Label.LEFT
        elif reward_1 < reward_2:
            return Label.RIGHT
        else:
            return Label.INDIFFERENT
