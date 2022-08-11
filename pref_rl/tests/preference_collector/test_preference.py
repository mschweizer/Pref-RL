from pref_rl.preference_collector.binary_choice import BinaryChoice
from pref_rl.preference_collector.preference import Preference
from pref_rl.query_generation.query import BinaryChoiceQuery


def test_equal():
    choice = BinaryChoice(0)
    query = BinaryChoiceQuery(["segment_1", "segment_2"])
    assert Preference(query, choice) == Preference(query, choice)


def test_not_equal():
    choice = BinaryChoice(0)
    choice_set = ["segment_1", "segment_2"]
    query_1 = BinaryChoiceQuery(choice_set)
    query_2 = BinaryChoiceQuery(choice_set)
    assert Preference(query_1, choice) != Preference(query_2, choice)


def test_string_repr():
    choice = BinaryChoice(0)
    pref = Preference(BinaryChoiceQuery(["segment_1", "segment_2"]), choice)
    assert str(pref) == "Choice: " + str(choice)
