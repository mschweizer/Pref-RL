from typing import List

from .querent import AbstractPreferenceQuerent
from .query_selection.selector import AbstractQuerySelector
from ..preference_data.query import Query


class DummyPreferenceQuerent(AbstractPreferenceQuerent):

    def __init__(self, query_selector: AbstractQuerySelector):
        """
        This preference querent only pretends to query the user. It just selects the specified number of queries from
        the list of candidate queries and returns these selected queries. This dummy querent is useful in combination
        when the agent does not interact with a (real) user but gets the preferences from an oracle instead.
        :param query_selector: The query selector that is used for selecting queries from the query candidates.
        """
        super(DummyPreferenceQuerent, self).__init__(query_selector)

    def query_preferences(self, query_candidates: List[Query], num_queries: int) -> List[Query]:
        """
        :param query_candidates: The list of query candidates.
        :param num_queries: The number of queries that should be answered by the user (or an oracle).
        :return: The list of selected queries.
        """
        return self.query_selector.select_queries(query_candidates, num_queries)