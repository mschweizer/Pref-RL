import random


class RandomQuerySelector:
    @staticmethod
    def select_query(queries):
        return random.choice(queries)
