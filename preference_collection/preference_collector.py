from abc import ABC, abstractmethod

from preference_collection.preference_oracle import RewardMaximizingOracleMixin
from query_selection.query_selector import AbstractQuerySelector, MostRecentlyGeneratedQuerySelector


# TODO: Rename to PreferenceQuerent and create additional PreferenceCollector (m: collect_preferences())
class AbstractPreferenceCollectorMixin(AbstractQuerySelector, ABC):

    # TODO: Make QuerySelector a component
    def __init__(self, preferences, query_candidates):
        self.query_candidates = query_candidates
        self.preferences = preferences

    # TODO: Change signature to query_preferences(query_candidates, num_queries)
    @abstractmethod
    def query_preferences(self, num_preferences):
        pass


# TODO: Rename into SynchronousPreferenceQuerent
class BaseSyntheticPreferenceCollectorMixin(AbstractPreferenceCollectorMixin,
                                            MostRecentlyGeneratedQuerySelector, RewardMaximizingOracleMixin):

    def query_preferences(self, num_preferences):
        queries = self.select_queries(self.query_candidates, num_queries=num_preferences)
        self.preferences.extend([(query, self.answer(query)) for query in queries])
