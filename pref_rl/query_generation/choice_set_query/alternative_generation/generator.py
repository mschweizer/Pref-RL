from abc import ABC, abstractmethod


class AbstractAlternativeGenerator(ABC):

    @abstractmethod
    def generate(self, policy_model, num_alternatives):
        pass
