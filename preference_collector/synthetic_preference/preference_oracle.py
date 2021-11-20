from abc import ABC, abstractmethod

from preference_collector.binary_choice import BinaryChoice


class AbstractOracle(ABC):

    @abstractmethod
    def answer(self, query):
        pass


class RewardMaximizingOracle(AbstractOracle):
    def answer(self, query):
        assert len(query.choice_set) == 2, \
            "Preference oracle assumes choice sets of size 2, but found {num} items.".format(num=len(query.choice_set))
        reward_1, reward_2 = self.compute_total_original_rewards(query)
        return self.compute_preference(reward_1, reward_2)

    @staticmethod
    def compute_total_original_rewards(query):
        return (sum(info["external_reward"] for info in segment.infos) for segment in query.choice_set)

    @staticmethod
    def compute_preference(reward_1, reward_2):
        if reward_1 > reward_2:
            return BinaryChoice.LEFT
        elif reward_1 < reward_2:
            return BinaryChoice.RIGHT
        else:
            return BinaryChoice.INDIFFERENT
