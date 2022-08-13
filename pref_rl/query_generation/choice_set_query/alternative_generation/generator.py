from abc import ABC, abstractmethod
from typing import List

from ....agents.policy.model import PolicyModel


class AbstractAlternativeGenerator(ABC):
    """
    The base for alternative generators. An alternative generator is responsible for generating the choice alternatives
    that are contained in choice set queries.
    """

    @abstractmethod
    def generate(self, policy_model: PolicyModel, num_alternatives: int) -> List:
        """
        Generates alternatives.
        :param policy_model: The policy model that is used to generate the alternatives.
        :param num_alternatives: The number of alternatives that are generated.
        :return: The list of generated alternatives.
        """
