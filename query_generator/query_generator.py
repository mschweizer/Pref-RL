from abc import ABC, abstractmethod


class AbstractQueryGenerator(ABC):

    @abstractmethod
    def generate_queries(self, policy_model, num_queries):
        pass
