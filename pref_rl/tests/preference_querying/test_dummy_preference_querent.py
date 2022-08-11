from ...preference_querying.dummy_querent import DummyPreferenceQuerent
from ...preference_querying.query_selection.selector import RandomQuerySelector


def test_queries_correct_number_of_queries():
    preference_querent = DummyPreferenceQuerent(query_selector=RandomQuerySelector())
    query_candidates = ["query1", "query2"]
    num_preference_queries = 1

    newly_pending_queries = preference_querent.query_preferences(query_candidates, num_queries=num_preference_queries)

    assert len(newly_pending_queries) == num_preference_queries
