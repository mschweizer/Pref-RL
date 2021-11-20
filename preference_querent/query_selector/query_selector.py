import itertools
import logging
import random
from abc import ABC, abstractmethod
from typing import List

from query_generator.query import Query


class AbstractQuerySelector(ABC):

    @abstractmethod
    def select_queries(self, query_candidates: List[Query], num_queries: int = 1) -> List[Query]:
        """ Returns a specified number of selected queries from the set of candidates. """


class RandomQuerySelector(AbstractQuerySelector):

    def select_queries(self, query_candidates, num_queries=1):
        try:
            return random.sample(query_candidates, num_queries)
        except ValueError as e:
            logging.warning(str(e) + " Returning empty set of queries.")
            return []


class MostRecentlyGeneratedQuerySelector(AbstractQuerySelector):

    def select_queries(self, query_candidates, num_queries=1):
        try:
            return list(itertools.islice(query_candidates, len(query_candidates) - num_queries, len(query_candidates)))
        except ValueError as e:
            logging.warning(str(e) + " Returning empty set of queries.")
            return []
