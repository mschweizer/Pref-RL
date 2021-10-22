from abc import ABC, abstractmethod
from typing import List


class AbstractPreferenceQuerent(ABC):
    def __init__(self, query_selector):
        self.query_selector = query_selector

    @abstractmethod
    def query_preferences(self, query_candidates, num_queries) -> List:
        pass


class DummyPreferenceQuerent(AbstractPreferenceQuerent):

    def __init__(self, query_selector, preference_collector):
        super(DummyPreferenceQuerent, self).__init__(query_selector)
        self.preference_collector = preference_collector

    def query_preferences(self, query_candidates, num_queries) -> List:
        newly_pending_queries = self.query_selector.select_queries(query_candidates, num_queries)
        self.preference_collector.pending_queries.extend(newly_pending_queries)
        self.preference_collector.collect_preferences()
        return []
