from abc import ABC, abstractmethod

from preference_data.querent.preference_querent import AbstractPreferenceQuerent


class AbstractSynchronousPreferenceQuerent(AbstractPreferenceQuerent, ABC):
    def query_preferences(self, queries):
        return [(query, self.answer(query)) for query in queries]

    @abstractmethod
    def answer(self, query):
        pass
