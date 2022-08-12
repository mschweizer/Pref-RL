import random
from typing import List, Tuple

from .generator import AbstractChoiceSetQueryGenerator
from ...utils.logging import create_logger


class RandomChoiceSetQueryGenerator(AbstractChoiceSetQueryGenerator):
    def __init__(self, alternative_generator, alternatives_per_choice_set=2):
        super().__init__(alternative_generator, alternatives_per_choice_set)
        self.logger = create_logger("RandomChoiceSetQueryGenerator")

    def select_choice_sets(self, num_choice_sets: int, num_alternatives_per_choice_set: int, alternatives) \
            -> List[Tuple]:
        return [self._select_alternatives(alternatives, num_alternatives_per_choice_set)
                for _ in range(num_choice_sets)]

    def _select_alternatives(self, alternatives, num_alternatives_per_choice_set) -> Tuple:
        try:
            return tuple(random.sample(alternatives, num_alternatives_per_choice_set))
        except ValueError as e:
            self.logger.warning(str(e) + " Returning empty sample.")
            return tuple()
