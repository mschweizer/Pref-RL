import logging
from abc import ABC, abstractmethod

from data_generation.preference_label import PreferenceLabel
from data_generation.query_selector import RandomQuerySelector


class PreferenceCollector(ABC):

    def __init__(self, queries):
        self.queries = queries
        self.query_selector = RandomQuerySelector()
        self.preferences = []

    def try_save_preference(self):
        try:
            self.save_preference()
        except IndexError as e:
            logging.warning("Preference collection failed. There are no preference queries available. "
                            "Original error message: " + str(e))

    def save_preference(self):
        query = self.query_selector.select_query(self.queries)
        preference = self.collect_preference(query)
        self.preferences.append((query, preference))

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
