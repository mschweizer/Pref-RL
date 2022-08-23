from collections import deque
from typing import List, Tuple, Any

import numpy as np

from .alternative_generation.generator import AbstractAlternativeGenerator
from .random_generator import RandomChoiceSetQueryGenerator
from ...utils.logging import get_or_create_logger

SAMPLING_RESULT_MSG = "Sampled alternative {}/{} as choice alternative {}/{}"

MOSTLY_NEW_ALTERNATIVES_MSG = "There are mostly new alternatives (> 80%) in the buffer."

BUFFER_TOO_SMALL_MSG = "The item buffer of size {size} is too small to fit {new} new alternatives. " \
                       "Using only new alternatives and no old alternatives in this iteration."


class BufferedChoiceSetQueryGenerator(RandomChoiceSetQueryGenerator):
    def __init__(self, alternative_generator: AbstractAlternativeGenerator,
                 alternatives_per_choice_set: int = 2,
                 buffer_size: int = 400):
        """
        This choice set generator buffers a limited number of alternatives from previous iterations and uses these
        buffered alternatives and newly generated alternatives to generate the queries.
        :param alternative_generator: The generator that generates the alternatives contained in the choice sets.
        :param alternatives_per_choice_set: The number of alternatives contained in each choice set.
        :param buffer_size: The size of the buffer used for buffering alternatives.
        """
        super(BufferedChoiceSetQueryGenerator, self).__init__(alternative_generator, alternatives_per_choice_set)
        self.buffer = deque(maxlen=buffer_size)
        self.logger = get_or_create_logger(self.__class__.__name__)

    def _select_choice_sets(self, num_choice_sets: int, new_alternatives) -> List[Tuple]:
        self.buffer.extend(new_alternatives)
        if self.buffer.maxlen > len(new_alternatives):
            self._warn_if_buffer_mostly_filled_with_new_alternatives(new_alternatives)
            choice_sets = self._select_choice_sets_from_new_and_buffered(num_choice_sets)
        else:
            choice_sets = self._select_choice_sets_only_from_new(num_choice_sets, new_alternatives)
        return choice_sets

    def _select_choice_sets_from_new_and_buffered(self, num_choice_sets: int) -> List[Tuple]:
        return [self._select_alternatives(list(self.buffer)) for _ in range(num_choice_sets)]

    def _warn_if_buffer_mostly_filled_with_new_alternatives(self, new_alternatives: List) -> None:
        if len(new_alternatives) > 0.8 * self.buffer.maxlen:
            self.logger.warning(MOSTLY_NEW_ALTERNATIVES_MSG)

    def _select_alternatives(self, buffered_alternatives: List) -> Tuple:
        probabilities, reversed_probabilities = self._compute_selection_probabilities(buffered_alternatives)

        selected_alternatives = []
        for i in range(self.alternatives_per_choice_set):
            while True:
                selected_alternative = \
                    self._select_alternative(buffered_alternatives, i, probabilities, reversed_probabilities)
                if selected_alternative not in selected_alternatives \
                        or len(buffered_alternatives) < self.alternatives_per_choice_set:  # duplicate-free not possible
                    selected_alternatives.append(selected_alternative)
                    break

        return tuple(selected_alternatives)

    @staticmethod
    def _compute_selection_probabilities(alternatives: List) -> Tuple[List, List]:
        """
        Newer alternatives (at the right end of the list) are assigned higher selection probabilities. Selection
        probabilities decay linearly in the list index of the alternative.
        :param alternatives: The list of alternatives over which the probability distribution for selection is defined.
        :return: The selection probability distribution p and the reversed distribution p_rev with p[i] = p_rev[n-i].
        """
        weights = [(i + 1) / len(alternatives) for i in range(len(alternatives))]
        probabilities = [weight / sum(weights) for weight in weights]
        reversed_probabilities = probabilities.copy()
        reversed_probabilities.reverse()
        return probabilities, reversed_probabilities

    def _select_alternative(self, alternatives: List, i: int, probabilities: List, reversed_probabilities: List) -> Any:
        """
        Alternatives are selected according to the given probability distribution p defined over the list of
        alternatives. Every other alternative is selected according to the reversed probability distribution p_rev.
        """
        if i % 2 == 0:
            idx = np.random.choice(range(len(alternatives)), p=probabilities)
            selected_alternative = alternatives[idx]
            self.logger.debug(SAMPLING_RESULT_MSG.format(idx, len(alternatives), i, self.alternatives_per_choice_set))
        else:
            idx = np.random.choice(range(len(alternatives)), p=reversed_probabilities)
            selected_alternative = np.random.choice(alternatives, p=reversed_probabilities)
            self.logger.debug(SAMPLING_RESULT_MSG.format(idx, len(alternatives), i, self.alternatives_per_choice_set))
        return selected_alternative

    def _select_choice_sets_only_from_new(self, num_choice_sets: int, new_alternatives: List) -> List[Tuple]:
        self.logger.warning(BUFFER_TOO_SMALL_MSG.format(size=self.buffer.maxlen, new=len(new_alternatives)))
        return [super(BufferedChoiceSetQueryGenerator, self)._select_alternatives(new_alternatives)
                for _ in range(num_choice_sets)]
