from abc import ABC, abstractmethod

from preference_collection.preference_oracle import AbstractOracle, RewardMaximizingOracle
from query_selection.query_selector import AbstractQuerySelector, MostRecentlyGeneratedQuerySelector


class AbstractPreferenceCollectorMixin(AbstractQuerySelector, ABC):

    def __init__(self, preferences, query_candidates):
        self.query_candidates = query_candidates
        self.preferences = preferences

    @abstractmethod
    def query_preferences(self, num_preferences):
        pass


class AbstractSyntheticPreferenceCollectorMixin(AbstractPreferenceCollectorMixin, AbstractOracle, ABC):

    def query_preferences(self, num_preferences):
        queries = self.select_queries(self.query_candidates, num_queries=num_preferences)
        self.preferences.extend([(query, self.answer(query)) for query in queries])


class SyntheticPreferenceCollectorMixin(AbstractSyntheticPreferenceCollectorMixin,
                                        MostRecentlyGeneratedQuerySelector, RewardMaximizingOracle):
    pass
