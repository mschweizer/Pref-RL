from unittest.mock import Mock

from preference_collection.preference_querent import DummyPreferenceQuerent
from query_selection.query_selector import RandomQuerySelector


def test_queries_correct_number_of_queries():
    preference_collector = Mock()
    preference_collector.collect_preferences = Mock()
    preference_querent = DummyPreferenceQuerent(query_selector=RandomQuerySelector(),
                                                preference_collector=preference_collector)
    query_candidates = ["query1", "query2"]
    num_preference_queries = 1

    preference_querent.query_preferences(query_candidates, num_queries=num_preference_queries)

    assert len(preference_collector.pending_queries.extend.call_args.args[0]) == num_preference_queries
