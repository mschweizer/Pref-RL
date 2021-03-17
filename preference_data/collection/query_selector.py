import random


class RandomQuerySelector:
    @staticmethod
    def select_query(queries):
        return random.choice(queries)

    def select_queries(self, queries, num_queries):
        return [self.select_query(queries) for _ in range(num_queries)]
