from unittest.mock import Mock

from preference_collection.label import Label
from preference_collection.preference_collector import SyntheticPreferenceCollectorMixin


def test_query_preferences():
    queries = [["seg1", "seg2"]]
    indifferent = Label.INDIFFERENT
    preferences = []
    preference_querent = SyntheticPreferenceCollectorMixin(query_candidates=queries, preferences=preferences)
    preference_querent.answer = Mock(return_value=indifferent)

    num_preferences = 1
    preference_querent.query_preferences(num_preferences=num_preferences)

    assert len(preferences) == num_preferences
    assert type(preferences[0]) is tuple
    assert preferences[0] == (queries[0], indifferent)
