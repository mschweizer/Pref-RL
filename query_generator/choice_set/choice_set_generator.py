import logging

from query_generator.query import ChoiceQuery
from query_generator.query_generator import AbstractQueryGenerator


class ChoiceSetGenerator(AbstractQueryGenerator):
    def __init__(self, item_generator, item_selector, items_per_query=2):
        self.item_generator = item_generator
        self.item_selector = item_selector
        self.items_per_query = items_per_query

    def generate_queries(self, policy_model, num_queries):
        num_items = self._calculate_num_items(num_queries)
        items = self.item_generator.generate(policy_model, num_items)
        queries = self._generate_queries(items, num_queries)
        return queries

    def _generate_queries(self, items, num_queries):
        queries = []
        for _ in range(num_queries):
            try:
                query = ChoiceQuery(choice_set=self.item_selector.select_items(items, num_items=self.items_per_query))
                queries.append(query)
            except AssertionError as e:
                logging.warning(str(e))
        return queries

    def _calculate_num_items(self, num_queries):
        if num_queries / self.items_per_query > 20:
            return int(num_queries / self.items_per_query)
        else:
            return int(num_queries * self.items_per_query)
