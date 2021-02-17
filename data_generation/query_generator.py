import random


class RandomQueryGenerator:
    def __init__(self, segment_samples, query_set_size=2):
        self.segment_samples = segment_samples
        self.query_set_size = query_set_size
        self.queries = []

    def save_query(self):
        query = self.generate_query()
        self.queries.append(query)

    def generate_query(self):
        return random.sample(self.segment_samples, self.query_set_size)
