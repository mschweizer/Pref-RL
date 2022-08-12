import random
from typing import Tuple

from .generator import AbstractChoiceSetQueryGenerator


class RandomChoiceSetQueryGenerator(AbstractChoiceSetQueryGenerator):
    def _select_alternatives(self, alternatives) -> Tuple:
        try:
            return tuple(random.sample(alternatives, self.alternatives_per_choice_set))
        except ValueError as e:
            self.logger.warning(str(e) + " Returning empty sample.")
            return tuple()