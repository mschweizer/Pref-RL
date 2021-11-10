from abc import ABC, abstractmethod


class AbstractQueryItemGenerator(ABC):

    @abstractmethod
    def generate(self, policy_model, num_items):
        pass
