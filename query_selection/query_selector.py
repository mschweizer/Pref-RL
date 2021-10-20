import itertools
import random
from abc import ABC, abstractmethod


class AbstractQuerySelector(ABC):

    @abstractmethod
    def select_queries(self, query_candidates, num_queries=1):
        pass


class RandomQuerySelector(AbstractQuerySelector):

    def select_queries(self, query_candidates, num_queries=1):
        return [self.select_query(query_candidates) for _ in range(num_queries)]

    @staticmethod
    def select_query(queries):
        return random.choice(queries)


class MostRecentlyGeneratedQuerySelector(AbstractQuerySelector):

    def select_queries(self, query_candidates, num_queries=1):
        return list(itertools.islice(query_candidates, len(query_candidates) - num_queries, len(query_candidates)))
