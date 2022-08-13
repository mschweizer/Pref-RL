import random
from typing import Tuple, List

from .alternative_generation.generator import AbstractAlternativeGenerator
from .generator import AbstractChoiceSetQueryGenerator
from ...utils.logging import create_logger


class RandomChoiceSetQueryGenerator(AbstractChoiceSetQueryGenerator):
    def __init__(self, alternative_generator: AbstractAlternativeGenerator, alternatives_per_choice_set: int = 2):
        """
        This choice set generator selects the specified number of alternatives from the generated alternatives uniformly
        at random.
        :param alternative_generator: The generator that generates the alternatives contained in the choice sets.
        :param alternatives_per_choice_set: The number of alternatives contained in each choice set.
        """
        super().__init__(alternative_generator, alternatives_per_choice_set)
        self.logger = create_logger(self.__class__.__name__)

    def _select_alternatives(self, alternatives: List) -> Tuple:
        try:
            return tuple(random.sample(alternatives, self.alternatives_per_choice_set))
        except ValueError as e:
            self.logger.warning(str(e) + " Returning empty sample.")
            return tuple()
