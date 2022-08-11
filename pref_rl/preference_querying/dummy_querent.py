from typing import List

from .querent import AbstractPreferenceQuerent


class DummyPreferenceQuerent(AbstractPreferenceQuerent):

    def __init__(self, query_selector):
        super(DummyPreferenceQuerent, self).__init__(query_selector)

    def query_preferences(self, query_candidates, num_queries) -> List:
        return self.query_selector.select_queries(query_candidates, num_queries)