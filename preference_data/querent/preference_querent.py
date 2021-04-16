from abc import ABC, abstractmethod


class AbstractPreferenceQuerent(ABC):

    @abstractmethod
    def query_preferences(self, queries):
        pass
