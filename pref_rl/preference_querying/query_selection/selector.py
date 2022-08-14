import itertools
import logging
import random
from abc import ABC, abstractmethod
from typing import List

from ...preference_data.query import Query


class AbstractQuerySelector(ABC):
    """
    The base for query selectors. A query selector is responsible for a specified number of queries from a list of query
    candidates.
    """

    @abstractmethod
    def select_queries(self, query_candidates: List[Query], num_queries: int = 1) -> List[Query]:
        """ Selects a specified number of queries from the list of candidates.
        :param query_candidates: The list of query candidates.
        :param num_queries: The specified number of queries to be selected.
        :return: The list of selected queries.
        """


class RandomQuerySelector(AbstractQuerySelector):
    """
    This query selector selects queries from the list of candidates uniformly at random.
    """

    def select_queries(self, query_candidates: List[Query], num_queries: int = 1) -> List[Query]:
        try:
            return random.sample(query_candidates, num_queries)
        except ValueError as e:
            logging.warning(str(e) + " Returning empty set of queries.")
            return []


class MostRecentlyGeneratedQuerySelector(AbstractQuerySelector):
    """
    This query selector selects from the list of candidates the most recently generated queries. The selector assumes
    that the most recently generated queries are rightmost in the candidate list.
    """

    def select_queries(self, query_candidates: List[Query], num_queries: int = 1) -> List[Query]:
        try:
            return list(itertools.islice(query_candidates, len(query_candidates) - num_queries, len(query_candidates)))
        except ValueError as e:
            logging.warning(str(e) + " Returning empty set of queries.")
            return []
