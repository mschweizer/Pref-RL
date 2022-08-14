from abc import ABC, abstractmethod
from typing import List

from .query_selection.selector import AbstractQuerySelector
from ..preference_data.query import Query


class AbstractPreferenceQuerent(ABC):
    """
    The base for preference querents. A preference querent is responsible for querying the user (sending queries to the
    user) in order to elicit the user's preferences.
    """

    def __init__(self, query_selector: AbstractQuerySelector):
        self.query_selector = query_selector

    @abstractmethod
    def query_preferences(self, query_candidates: List[Query], num_queries: int) -> List[Query]:
        """
        Sends a specified number of queries from a list of candidate queries to the user.
        :param query_candidates: The list of query candidates.
        :param num_queries: The number of queries that should be sent to the user.
        :return: The list of queries that were actually sent to the user.
        """
