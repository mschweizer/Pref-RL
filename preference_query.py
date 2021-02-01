import random
from abc import ABC, abstractmethod

from trajectory_sampling import TrajectorySegmentSampler


class PreferenceDataGenerator:

    def __init__(self, trajectory_buffer):
        self.segment_sampler = \
            TrajectorySegmentSampler(trajectory_buffer=trajectory_buffer, segment_length=10)  # TODO: make init param
        self.query_generator = RandomQueryGenerator()
        self.preference_collector = RewardMaximizingPreferenceCollector()

        self.segment_samples = []
        self.queries = []
        self.preferences = []

    def generate_sample(self):
        sample = self.segment_sampler.generate_sample()
        self.segment_samples.append(sample)

    def generate_query(self):
        query = self.query_generator.generate_query(self.segment_samples)
        self.queries.append(query)

    def collect_preference(self):
        preference = self.preference_collector.collect_preference(self.queries)
        self.preferences.append(preference)


class RandomQuerySelector:
    @staticmethod
    def select_query(queries):
        return random.choice(queries)


class PreferenceCollector(ABC):

    def __init__(self):
        self.query_selector = RandomQuerySelector()

    @abstractmethod
    def collect_preference(self, query):
        """Returns query elements in preferred order (from most to least preferred)."""


class RandomPreferenceCollector(PreferenceCollector):

    def collect_preference(self, queries):
        query = self.query_selector.select_query(queries)
        random.shuffle(query)
        return query


class RewardMaximizingPreferenceCollector(PreferenceCollector):

    def collect_preference(self, queries):
        query = self.query_selector.select_query(queries)

        segment_1 = query[0]
        segment_2 = query[1]

        reward_1 = sum(experience.info["original_reward"] for experience in segment_1)
        reward_2 = sum(experience.info["original_reward"] for experience in segment_2)

        return [segment_1, segment_2] if reward_1 >= reward_2 else [segment_2, segment_1]


class RandomQueryGenerator:
    def __init__(self, query_set_size=2):
        self.query_set_size = query_set_size

    def generate_query(self, segment_samples):
        return random.sample(segment_samples, self.query_set_size)
