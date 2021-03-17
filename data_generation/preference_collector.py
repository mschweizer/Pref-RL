from abc import ABC, abstractmethod

from data_generation.preference_label import PreferenceLabel
from reward_modeling.preference_dataset import PreferenceDataset


class PreferenceCollector(ABC):

    def __init__(self, queries, num_preferences=3000):
        self.queries = queries
        self.preferences = PreferenceDataset(capacity=num_preferences)

    def collect_preferences(self, queries):
        self.preferences.extend([(query, self.collect_preference(query)) for query in queries])

    @abstractmethod
    def collect_preference(self, query):
        """Returns preference (enum)"""


class RandomPreferenceCollector(PreferenceCollector):

    def collect_preference(self, query):
        return PreferenceLabel.random()


class RewardMaximizingPreferenceCollector(PreferenceCollector):

    def collect_preference(self, query):
        reward_1, reward_2 = self.compute_total_rewards(query)
        return self.compute_preference(reward_1, reward_2)

    @staticmethod
    def compute_total_rewards(query):
        return (sum(experience.info["original_reward"] for experience in segment) for segment in query)

    @staticmethod
    def compute_preference(reward_1, reward_2):
        if reward_1 > reward_2:
            return PreferenceLabel.LEFT
        elif reward_1 < reward_2:
            return PreferenceLabel.RIGHT
        else:
            return PreferenceLabel.INDIFFERENT
