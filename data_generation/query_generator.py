import random


class RandomQueryGenerator:
    def __init__(self, query_set_size=2):
        self.query_set_size = query_set_size

    def generate_query(self, segment_samples):
        return random.sample(segment_samples, self.query_set_size)
