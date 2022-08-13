from pref_rl.preference_data.binary_choice import BinaryChoice
from pref_rl.preference_data.preference import Preference
from pref_rl.preference_data.query import BinaryChoiceSetQuery


def test_equal():
    choice = BinaryChoice(0)
    query = BinaryChoiceSetQuery(["segment_1", "segment_2"])
    assert Preference(query, choice) == Preference(query, choice)


def test_not_equal():
    choice = BinaryChoice(0)
    choice_set = ["segment_1", "segment_2"]
    query_1 = BinaryChoiceSetQuery(choice_set)
    query_2 = BinaryChoiceSetQuery(choice_set)
    assert Preference(query_1, choice) != Preference(query_2, choice)


def test_string_repr():
    choice = BinaryChoice(0)
    pref = Preference(BinaryChoiceSetQuery(["segment_1", "segment_2"]), choice)
    assert str(pref) == "Choice: " + str(choice)
