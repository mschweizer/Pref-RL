import random
from abc import ABC, abstractmethod
from typing import Tuple


class AbstractQueryItemSelector(ABC):
    @abstractmethod
    def select_items(self, items, num_items) -> Tuple:
        pass


class RandomItemSelector(AbstractQueryItemSelector):
    def select_items(self, items, num_items) -> Tuple:
        return tuple(random.sample(items, num_items))
