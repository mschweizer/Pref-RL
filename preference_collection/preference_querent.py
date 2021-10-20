from abc import ABC, abstractmethod
from typing import List


class AbstractPreferenceQuerent(ABC):
    def __init__(self, query_selector):
        self.query_selector = query_selector

    @abstractmethod
    def query_preferences(self, query_candidates, num_queries) -> List:
        pass


class DummyPreferenceQuerent(AbstractPreferenceQuerent):

    def __init__(self, query_selector):
        super(DummyPreferenceQuerent, self).__init__(query_selector)

    def query_preferences(self, query_candidates, num_queries) -> List:
        return self.query_selector.select_queries(query_candidates, num_queries)
