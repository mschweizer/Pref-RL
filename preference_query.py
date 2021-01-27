import random
from abc import ABC, abstractmethod


class PreferenceDataGenerator:

    def __init__(self, query_collector, trajectory_segment_samples, preference_data):
        self.trajectory_segment_samples = trajectory_segment_samples
        self.preference_data = preference_data
        self.query_collector = query_collector

    def collect_preference(self):
        query = self.generate_query()
        preference = self.query_collector.query_answer(query)
        self.preference_data.append(preference)

    def generate_query(self):
        return random.sample(self.trajectory_segment_samples, 2)


class PreferenceCollector(ABC):

    @abstractmethod
    def query_answer(self, query):
        """Returns query elements in preferred order (from most to least preferred)."""


class RandomPreferenceCollector(PreferenceCollector):

    def query_answer(self, query):
        query_shuffle_copy = query.copy()
        random.shuffle(query_shuffle_copy)
        return query_shuffle_copy


class RewardMaximizingPreferenceCollector(PreferenceCollector):

    def query_answer(self, query):
        segment_1 = query[0]
        segment_2 = query[1]

        reward_1 = sum(experience.info["original_reward"] for experience in segment_1)
        reward_2 = sum(experience.info["original_reward"] for experience in segment_2)

        return [segment_1, segment_2] if reward_1 >= reward_2 else [segment_2, segment_1]
