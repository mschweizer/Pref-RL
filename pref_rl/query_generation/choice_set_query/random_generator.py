import logging
import random
from typing import List, Tuple

from pref_rl.query_generation.choice_set_query.generator import AbstractChoiceSetQueryGenerator


class RandomChoiceSetQueryGenerator(AbstractChoiceSetQueryGenerator):
    def select_choice_sets(self, num_choice_sets: int, num_alternatives_per_choice_set: int, alternatives) \
            -> List[Tuple]:
        return [self._select_alternatives(alternatives, num_alternatives_per_choice_set)
                for _ in range(num_choice_sets)]

    @staticmethod
    def _select_alternatives(alternatives, num_alternatives) -> Tuple:
        try:
            return tuple(random.sample(alternatives, num_alternatives))
        except ValueError as e:
            logging.warning(str(e) + " Returning empty sample.")
            return tuple()