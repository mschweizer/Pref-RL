from preference_collection.preference_querent import DummyPreferenceQuerent
from query_selection.query_selector import RandomQuerySelector


def test_queries_correct_number_of_queries():
    preference_querent = DummyPreferenceQuerent(query_selector=RandomQuerySelector())
    query_candidates = ["query1", "query2"]
    num_preference_queries = 1

    newly_pending_queries = preference_querent.query_preferences(query_candidates, num_queries=num_preference_queries)

    assert len(newly_pending_queries) == num_preference_queries
