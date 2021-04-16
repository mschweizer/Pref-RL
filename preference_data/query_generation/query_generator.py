from abc import ABC, abstractmethod


class AbstractQueryGenerator(ABC):

    @abstractmethod
    def generate_queries(self, num_queries=1, with_training=True):
        pass
