import logging
from abc import ABC, abstractmethod
from typing import List, Tuple

from .alternative_generation.generator import AbstractAlternativeGenerator
from ..generator import AbstractQueryGenerator
from ...agents.policy.model import PolicyModel
from ...preference_data.query import ChoiceSetQuery


class AbstractChoiceSetQueryGenerator(AbstractQueryGenerator, ABC):
    def __init__(self, alternative_generator: AbstractAlternativeGenerator, alternatives_per_choice_set: int = 2):
        """
        A choice set query generator generates choice set queries: a special type of preference query that asks the
        user to make a choice among a finite set of alternatives.
        It generates these queries by selecting the specified number of alternatives from a set of candidates for each
        query. It generates these alternatives in the first place.
        :param alternative_generator: The generator that generates the alternatives contained in the choice sets.
        :param alternatives_per_choice_set: The number of alternatives contained in each choice set.
        """
        self.alternative_generator = alternative_generator
        self.alternatives_per_choice_set = alternatives_per_choice_set

    def generate_queries(self, policy_model: PolicyModel, num_queries: int) -> List[ChoiceSetQuery]:
        """
        :param policy_model: The policy model that is used to generate the queries.
        :param num_queries: The number of queries that are generated.
        :return: The list of generated queries.
        """
        num_candidate_alternatives = self._calculate_num_alternatives(num_queries)
        candidate_alternatives = self.alternative_generator.generate(policy_model, num_candidate_alternatives)
        choice_sets = self._select_choice_sets(num_queries, candidate_alternatives)
        queries = []
        for choice_set in choice_sets:
            try:
                query = ChoiceSetQuery(choice_set)
                queries.append(query)
            except AssertionError as e:
                logging.warning(str(e))
        return queries

    def _calculate_num_alternatives(self, num_queries: int) -> int:
        if num_queries / self.alternatives_per_choice_set > 20:
            return int(num_queries / self.alternatives_per_choice_set)
        else:
            return int(num_queries * self.alternatives_per_choice_set)

    def _select_choice_sets(self, num_choice_sets: int, alternatives: List) -> List[Tuple]:
        return [self._select_alternatives(alternatives) for _ in range(num_choice_sets)]

    @abstractmethod
    def _select_alternatives(self, alternatives: List) -> Tuple:
        raise NotImplementedError
