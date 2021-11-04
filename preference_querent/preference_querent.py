from abc import ABC, abstractmethod
from typing import List


class AbstractPreferenceQuerent(ABC):
    def __init__(self, query_selector):
        self.query_selector = query_selector

    @abstractmethod
    def query_preferences(self, query_candidates, num_queries) -> List:
        pass


class SynchronousPreferenceQuerent(AbstractPreferenceQuerent):

    def __init__(self, query_selector, preference_collector, preferences):
        super(SynchronousPreferenceQuerent, self).__init__(query_selector)
        self.preference_collector = preference_collector
        self.preferences = preferences

    def query_preferences(self, query_candidates, num_queries) -> List:
        newly_pending_queries = self.query_selector.select_queries(query_candidates, num_queries)
        self.preference_collector.pending_queries.extend(newly_pending_queries)
        just_collected_preferences = self.preference_collector.collect_preferences()
        self.preferences.extend(just_collected_preferences)
        return []
