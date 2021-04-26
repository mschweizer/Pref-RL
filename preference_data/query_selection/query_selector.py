import random
from abc import ABC, abstractmethod


class AbstractQuerySelector(ABC):

    @abstractmethod
    def select_queries(self, queries, num_queries=1):
        pass


class RandomQuerySelector(AbstractQuerySelector):

    def select_queries(self, queries, num_queries=1):
        return [self.select_query(queries) for _ in range(num_queries)]

    @staticmethod
    def select_query(queries):
        return random.choice(queries)


class IndexQuerySelector(AbstractQuerySelector):

    def select_queries(self, queries, num_queries=1):
        return queries[:num_queries]
