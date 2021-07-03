from abc import ABC, abstractmethod

from preference_data.preference.label import Label
from preference_data.querent.preference_querent import AbstractPreferenceQuerent


class AbstractOracle(AbstractPreferenceQuerent, ABC):
    def query_preferences(self, queries):
        return [(query, self.answer(query)) for query in queries]

    @abstractmethod
    def answer(self, query):
        pass


class OriginalRewardMaximizingOracle(AbstractOracle):
    def answer(self, query):
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


class RandomOracle(AbstractOracle):
    def answer(self, query):
        return Label.random()
