from unittest.mock import Mock

from ....query_generation.choice_set_query.buffered_generator import BufferedChoiceSetQueryGenerator, \
    MOSTLY_NEW_ALTERNATIVES_MSG, BUFFER_TOO_SMALL_MSG


def test_buffer_is_filled_with_new_alternatives():
    buffered_generator = BufferedChoiceSetQueryGenerator(alternative_generator=Mock(), buffer_size=4)

    alternatives = [1, 2, 3, 4, 5]
    buffered_generator._select_choice_sets(1, new_alternatives=alternatives)

    assert alternatives[-buffered_generator.buffer.maxlen:] == list(buffered_generator.buffer)


def test_selects_alternatives_from_buffer_and_new_alternatives():
    buffered_generator = BufferedChoiceSetQueryGenerator(alternative_generator=Mock(), buffer_size=10)

    new_alternatives = [1]
    buffered_alternatives = [6]
    buffered_generator.buffer.extend(buffered_alternatives)

    choice_sets = buffered_generator._select_choice_sets(num_choice_sets=1, new_alternatives=new_alternatives)

    for choice_set in choice_sets:
        for alternative in choice_set:
            assert alternative in new_alternatives + buffered_alternatives


def test_falls_back_to_all_new_alternatives_if_buffer_is_too_small():
    buffered_generator = BufferedChoiceSetQueryGenerator(alternative_generator=Mock(), buffer_size=4)
    new_alternatives = [i for i in range(buffered_generator.buffer.maxlen + 1)]

    selected_choice_sets = buffered_generator._select_choice_sets(num_choice_sets=1, new_alternatives=new_alternatives)
    for choice_set in selected_choice_sets:
        for alternative in choice_set:
            assert alternative in new_alternatives


def test_warns_if_buffer_is_mostly_filled_with_new_alternatives(caplog):
    buffered_generator = BufferedChoiceSetQueryGenerator(alternative_generator=Mock(), buffer_size=10)
    new_alternatives = [i for i in range(int(0.9 * buffered_generator.buffer.maxlen))]
    buffered_generator._select_choice_sets(num_choice_sets=1, new_alternatives=new_alternatives)

    assert caplog.records[0].levelname == "WARNING"
    assert MOSTLY_NEW_ALTERNATIVES_MSG in caplog.records[0].message


def test_warns_when_ignoring_buffer_because_it_is_too_small(caplog):
    buffered_generator = BufferedChoiceSetQueryGenerator(alternative_generator=Mock(), buffer_size=10)
    new_alternatives = [i for i in range(buffered_generator.buffer.maxlen + 1)]
    buffered_generator._select_choice_sets(num_choice_sets=1, new_alternatives=new_alternatives)

    assert caplog.records[0].levelname == "WARNING"
    assert BUFFER_TOO_SMALL_MSG.format(size=buffered_generator.buffer.maxlen, new=len(new_alternatives)) \
           in caplog.records[0].message


def test_selects_correct_number_of_alternatives():
    num_alternatives = 3
    buffered_generator = BufferedChoiceSetQueryGenerator(alternative_generator=Mock(), buffer_size=10,
                                                         alternatives_per_choice_set=num_alternatives)
    alternatives = [1, 2, 3, 4]
    selected_alternatives = buffered_generator._select_alternatives(alternatives)

    assert len(selected_alternatives) == num_alternatives


def test_selects_duplicates_when_necessary():
    alternatives_per_choice_set = 2
    buffered_generator = BufferedChoiceSetQueryGenerator(alternative_generator=Mock(), buffer_size=10,
                                                         alternatives_per_choice_set=alternatives_per_choice_set)
    alternatives = [i for i in range(alternatives_per_choice_set - 1)]
    selected_alternatives = buffered_generator._select_alternatives(alternatives)

    assert len(selected_alternatives) == alternatives_per_choice_set
    assert selected_alternatives[0] == selected_alternatives[1]


def test_selects_no_duplicates_when_enough_alternatives_exist():
    alternatives_per_choice_set = 2
    buffered_generator = BufferedChoiceSetQueryGenerator(alternative_generator=Mock(), buffer_size=10,
                                                         alternatives_per_choice_set=alternatives_per_choice_set)
    alternatives = [i for i in range(alternatives_per_choice_set)]

    for _ in range(100):
        selected_alternatives = buffered_generator._select_alternatives(alternatives)
        assert selected_alternatives[0] != selected_alternatives[1]


def test_computed_selection_probabilities_sum_to_1():
    probabilities = \
        BufferedChoiceSetQueryGenerator(alternative_generator=Mock())._compute_selection_probabilities([1, 2, 3])
    assert sum(probabilities[0]) == 1


def test_reversed_probabilities_are_correct():
    probabilities = \
        BufferedChoiceSetQueryGenerator(alternative_generator=Mock())._compute_selection_probabilities([1, 2, 3])
    probabilities[1].reverse()
    assert probabilities[0] == probabilities[1]
