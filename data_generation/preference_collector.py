import random
from abc import ABC, abstractmethod


class PreferenceCollector(ABC):

    @abstractmethod
    def collect_preference(self, query):
        """Returns query elements in preferred order (from most to least preferred)."""


class RandomPreferenceCollector(PreferenceCollector):

    def collect_preference(self, query):
        random.shuffle(query)
        return query


class RewardMaximizingPreferenceCollector(PreferenceCollector):

    def collect_preference(self, query):
        segment_1 = query[0]
        segment_2 = query[1]

        reward_1 = sum(experience.info["original_reward"] for experience in segment_1)
        reward_2 = sum(experience.info["original_reward"] for experience in segment_2)

        return [segment_1, segment_2] if reward_1 >= reward_2 else [segment_2, segment_1]
