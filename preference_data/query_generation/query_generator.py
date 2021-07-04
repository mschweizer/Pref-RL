from abc import ABC, abstractmethod


class AbstractQueryGenerator(ABC):

    def __init__(self, query_candidates):
        self.query_candidates = query_candidates

    @abstractmethod
    def generate_queries(self, num_queries=1, with_policy_training=True):
        pass
