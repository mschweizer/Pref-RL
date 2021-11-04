from unittest.mock import Mock

from query_generator.choice_set.choice_set_generator import ChoiceSetGenerator
from query_generator.query_item_selector import RandomItemSelector


def test_generates_correct_number_of_queries():
    item_generator = Mock()
    item_selector = Mock(**{'select_items.return_value': ["segment_1", "segment_2"]})
    choice_set_generator = ChoiceSetGenerator(item_generator, item_selector, items_per_query=2)
    num_queries = 3

    queries = choice_set_generator.generate_queries(policy_model=Mock(), num_queries=num_queries)

    assert len(queries) == num_queries


def test_choice_sets_have_correct_size():
    item_generator = Mock()
    item_generator.generate = Mock(return_value=[1, 2, 3, 4, 5, 6])

    item_selector = RandomItemSelector()
    items_per_query = 2

    choice_set_generator = ChoiceSetGenerator(item_generator, item_selector, items_per_query)

    queries = choice_set_generator.generate_queries(policy_model=Mock(), num_queries=1)

    assert len(queries[0].choice_set) == items_per_query


def test_calculate_num_items():
    query_generator = ChoiceSetGenerator(item_selector=Mock(), item_generator=Mock())
    num_samples = query_generator._calculate_num_items(num_queries=500)
    assert num_samples == 157


def test_calculate_num_items_for_one_query():
    num_query_items = 2
    query_generator = ChoiceSetGenerator(item_selector=Mock(), item_generator=Mock(),
                                         items_per_query=num_query_items)
    num_samples = query_generator._calculate_num_items(num_queries=1)
    assert num_samples == num_query_items
