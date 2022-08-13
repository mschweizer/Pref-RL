from abc import ABC, abstractmethod
from typing import List

from ..agents.policy.model import PolicyModel
from ..preference_data.query import Query


class AbstractQueryGenerator(ABC):
    """
    The base for query generators. A query generator is responsible for generating preference queries that are used to
    elicit the user's preferences in the preference-based RL process.
    """

    @abstractmethod
    def generate_queries(self, policy_model: PolicyModel, num_queries: int) -> List[Query]:
        """
        Generates preference queries.
        :param policy_model: The policy model that is used to generate the queries.
        :param num_queries: The number of queries that are generated.
        :return: The list of generated queries.
        """
