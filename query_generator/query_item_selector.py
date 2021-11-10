import logging
import random
from abc import ABC, abstractmethod
from typing import Tuple


class AbstractQueryItemSelector(ABC):
    @abstractmethod
    def select_items(self, items, num_items) -> Tuple:
        pass


class RandomItemSelector(AbstractQueryItemSelector):
    def select_items(self, items, num_items) -> Tuple:
        try:
            return tuple(random.sample(items, num_items))
        except ValueError as e:
            logging.warning(str(e) + " Returning empty sample.")
            return tuple()
