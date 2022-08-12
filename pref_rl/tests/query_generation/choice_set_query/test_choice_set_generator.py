from unittest.mock import Mock

from ....query_generation.choice_set_query.generator import AbstractChoiceSetQueryGenerator


class ConcreteChoiceSetQueryGenerator(AbstractChoiceSetQueryGenerator):
    def select_choice_sets(self, num_choice_sets: int, num_alternatives_per_choice_set: int, alternatives):
        return [("segment_1", "segment_2"), ("segment_1", "segment_2"), ("segment_1", "segment_2")]


def test_generates_correct_number_of_queries():
    alternative_generator = Mock()
    choice_set_generator = ConcreteChoiceSetQueryGenerator(alternative_generator, alternatives_per_choice_set=2)
    num_queries = 3

    queries = choice_set_generator.generate_queries(policy_model=Mock(), num_queries=num_queries)

    assert len(queries) == num_queries


def test_choice_sets_have_correct_size():
    alternative_generator = Mock()
    items_per_query = 2
    choice_set_generator = ConcreteChoiceSetQueryGenerator(alternative_generator, items_per_query)

    queries = choice_set_generator.generate_queries(policy_model=Mock(), num_queries=1)

    assert len(queries[0].choice_set) == items_per_query


def test_calculate_num_items():
    items_per_query = 2
    num_queries = 500
    query_generator = ConcreteChoiceSetQueryGenerator(alternative_generator=Mock(),
                                                      alternatives_per_choice_set=items_per_query)
    num_samples = query_generator._calculate_num_alternatives(num_queries=num_queries)
    assert num_samples == num_queries / items_per_query


def test_calculate_num_items_for_one_query():
    num_query_items = 2
    query_generator = ConcreteChoiceSetQueryGenerator(alternative_generator=Mock(),
                                                      alternatives_per_choice_set=num_query_items)
    num_samples = query_generator._calculate_num_alternatives(num_queries=1)
    assert num_samples == num_query_items
