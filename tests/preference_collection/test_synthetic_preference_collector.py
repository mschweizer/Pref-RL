from unittest.mock import Mock

from preference_collection.preference_collector import SyntheticPreferenceCollector


def test_answers_all_pending_queries():
    oracle = Mock()
    preference_collector = SyntheticPreferenceCollector(oracle)

    pending_queries = ["query1", "query2", "query3"]
    preference_collector.pending_queries = pending_queries

    preferences = preference_collector.collect_preferences()

    assert len(preferences) == len(pending_queries)
