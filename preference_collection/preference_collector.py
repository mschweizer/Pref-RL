from abc import ABC, abstractmethod

from preference_collection.preference_oracle import RewardMaximizingOracleMixin
from query_selection.query_selector import AbstractQuerySelectorMixin, MostRecentlyGeneratedQuerySelectorMixin


class AbstractPreferenceCollectorMixin(AbstractQuerySelectorMixin, ABC):

    def __init__(self, preferences, query_candidates):
        self.query_candidates = query_candidates
        self.preferences = preferences

    @abstractmethod
    def query_preferences(self, num_preferences):
        pass


class BaseSyntheticPreferenceCollectorMixin(AbstractPreferenceCollectorMixin,
                                            MostRecentlyGeneratedQuerySelectorMixin, RewardMaximizingOracleMixin):

    def query_preferences(self, num_preferences):
        queries = self.select_queries(self.query_candidates, num_queries=num_preferences)
        self.preferences.extend([(query, self.answer(query)) for query in queries])
