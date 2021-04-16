import random
from abc import ABC, abstractmethod


class AbstractQuerySelector(ABC):

    def select_queries(self, queries, num_queries=1):
        return [self.select_query(queries) for _ in range(num_queries)]

    @staticmethod
    @abstractmethod
    def select_query(queries):
        pass


class RandomQuerySelector(AbstractQuerySelector):

    @staticmethod
    def select_query(queries):
        return random.choice(queries)
