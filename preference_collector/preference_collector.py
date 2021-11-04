from abc import ABC, abstractmethod
from typing import List


class AbstractPreferenceCollector(ABC):

    def __init__(self):
        self.pending_queries = []

    @abstractmethod
    def collect_preferences(self) -> List:
        pass
