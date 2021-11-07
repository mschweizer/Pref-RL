from abc import ABC, abstractmethod

from preference_collection.label import Label


class AbstractOracleMixin(ABC):

    @abstractmethod
    def answer(self, query):
        pass


class RewardMaximizingOracleMixin(AbstractOracleMixin):
    def answer(self, query):
        reward_1, reward_2 = self.compute_total_original_rewards(query)
        return self.compute_preference(reward_1, reward_2)

    @staticmethod
    def compute_total_original_rewards(query):
        return (sum(experience.info["external_reward"] for experience in segment) for segment in query)

    @staticmethod
    def compute_preference(reward_1, reward_2):
        if reward_1 > reward_2:
            return Label.LEFT
        elif reward_1 < reward_2:
            return Label.RIGHT
        else:
            return Label.INDIFFERENT


class RandomOracleMixin(AbstractOracleMixin):
    def answer(self, query):
        return Label.random()
