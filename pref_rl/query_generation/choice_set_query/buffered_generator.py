from collections import deque
from typing import List, Tuple

import numpy as np

from .random_generator import RandomChoiceSetQueryGenerator
from ...utils.logging import create_logger

MOSTLY_NEW_ALTERNATIVES_MSG = "There are mostly new alternatives (> 80%) in the buffer."

BUFFER_TOO_SMALL_MSG = "The item buffer of size {size} is too small to fit {new_alternatives} new alternatives. " \
                       "Using only new alternatives and no old alternatives in this iteration."


class BufferedChoiceSetQueryGenerator(RandomChoiceSetQueryGenerator):
    def __init__(self, alternative_generator, alternatives_per_choice_set=2, buffer_size=400):
        super(BufferedChoiceSetQueryGenerator, self).__init__(alternative_generator, alternatives_per_choice_set)
        self.buffer = deque(maxlen=buffer_size)
        self.logger = create_logger(self.__class__.__name__)

    def select_choice_sets(self, num_choice_sets: int, alternatives) -> List[Tuple]:
        self.buffer.extend(alternatives)
        if self.buffer.maxlen > len(alternatives):
            choice_sets = self._select_from_new_and_buffered(alternatives, num_choice_sets)
        else:
            choice_sets = self._select_only_from_new(alternatives, num_choice_sets)
        return choice_sets

    def _select_from_new_and_buffered(self, alternatives, num_choice_sets):
        self._warn_if_buffer_mostly_filled_with_new_alternatives(alternatives)
        choice_sets = [self._select_alternatives(list(self.buffer)) for _ in range(num_choice_sets)]
        return choice_sets

    def _warn_if_buffer_mostly_filled_with_new_alternatives(self, alternatives):
        if len(alternatives) > 0.8 * self.buffer.maxlen:
            self.logger.warning(MOSTLY_NEW_ALTERNATIVES_MSG)

    def _select_alternatives(self, alternatives):
        probabilities, reversed_probabilities = self._compute_selection_probabilities(alternatives)

        selected_alternatives = []
        for i in range(self.alternatives_per_choice_set):
            while True:
                selected_alternative = self._select_alternative(alternatives, i, probabilities, reversed_probabilities)
                if selected_alternative not in selected_alternatives \
                        or len(alternatives) < self.alternatives_per_choice_set:  # duplicate-free not possible
                    selected_alternatives.append(selected_alternative)
                    break

        return selected_alternatives

    @staticmethod
    def _compute_selection_probabilities(alternatives):
        weights = [(i + 1) / len(alternatives) for i in range(len(alternatives))]
        probabilities = [weight / sum(weights) for weight in weights]
        reversed_probabilities = probabilities.copy()
        reversed_probabilities.reverse()
        return probabilities, reversed_probabilities

    @staticmethod
    def _select_alternative(alternatives, i, probabilities, reversed_probabilities):
        if i % 2 == 0:
            selected_alternative = np.random.choice(alternatives, p=probabilities)
        else:
            selected_alternative = np.random.choice(alternatives, p=reversed_probabilities)
        return selected_alternative

    def _select_only_from_new(self, alternatives, num_choice_sets):
        self.logger.warning(BUFFER_TOO_SMALL_MSG.format(size=self.buffer.maxlen, new_alternatives=len(alternatives)))
        return [super(BufferedChoiceSetQueryGenerator, self)._select_alternatives(alternatives)
                for _ in range(num_choice_sets)]
