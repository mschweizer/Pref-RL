from data_generation.preference_collector import RewardMaximizingPreferenceCollector
from data_generation.query_generator import RandomQueryGenerator
from data_generation.query_selector import RandomQuerySelector
from data_generation.trajectory_sampling import TrajectorySegmentSampler


class PreferenceDataGenerator:

    def __init__(self, trajectory_buffer, segment_length=10):
        self.segment_sampler = \
            TrajectorySegmentSampler(trajectory_buffer=trajectory_buffer, segment_length=segment_length)
        self.query_generator = RandomQueryGenerator()
        self.query_selector = RandomQuerySelector()
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
        query = self.query_selector.select_query(self.queries)
        preference = self.preference_collector.collect_preference(query)
        self.preferences.append(preference)
