import random
from abc import ABC, abstractmethod


class AbstractQueryItemSelector(ABC):
    @abstractmethod
    def select_items(self, items, num_items):
        pass


class RandomQueryItemSelector(AbstractQueryItemSelector):
    def select_items(self, items, num_items):
        return random.sample(items, num_items)
