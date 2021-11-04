from unittest.mock import Mock

from preference_collector.synthetic_preference.synthetic_preference_collector import SyntheticPreferenceCollector


def test_answers_all_pending_queries():
    oracle = Mock()
    preference_collector = SyntheticPreferenceCollector(oracle)

    pending_queries = ["query1", "query2", "query3"]
    preference_collector.pending_queries = pending_queries.copy()

    preferences = preference_collector.collect_preferences()

    assert len(preferences) == len(pending_queries)


def test_answered_queries_are_not_pending_anymore():
    oracle = Mock()
    preference_collector = SyntheticPreferenceCollector(oracle)

    preference_collector.pending_queries = ["query1", "query2", "query3"]

    preference_collector.collect_preferences()

    assert len(preference_collector.pending_queries) == 0
