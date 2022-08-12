import logging
from abc import ABC, abstractmethod
from typing import List, Tuple

from ..generator import AbstractQueryGenerator
from ...preference_data.query import ChoiceQuery


class AbstractChoiceSetQueryGenerator(AbstractQueryGenerator, ABC):
    def __init__(self, alternative_generator, alternatives_per_choice_set=2):
        self.alternative_generator = alternative_generator
        self.alternatives_per_choice_set = alternatives_per_choice_set

    def generate_queries(self, policy_model, num_queries):
        num_alternatives = self._calculate_num_alternatives(num_queries)
        alternatives = self.alternative_generator.generate(policy_model, num_alternatives)
        choice_sets = self.select_choice_sets(num_queries, self.alternatives_per_choice_set, alternatives)
        queries = []
        for choice_set in choice_sets:
            try:
                query = ChoiceQuery(choice_set)
                queries.append(query)
            except AssertionError as e:
                logging.warning(str(e))
        return queries

    def _calculate_num_alternatives(self, num_queries):
        if num_queries / self.alternatives_per_choice_set > 20:
            return int(num_queries / self.alternatives_per_choice_set)
        else:
            return int(num_queries * self.alternatives_per_choice_set)

    @abstractmethod
    def select_choice_sets(self, num_choice_sets: int, num_alternatives_per_choice_set: int, alternatives) \
            -> List[Tuple]:
        raise NotImplementedError
