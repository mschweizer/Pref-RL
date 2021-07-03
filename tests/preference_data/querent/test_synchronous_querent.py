from unittest.mock import patch

from preference_data.querent.oracle import AbstractOracle


@patch.multiple(AbstractOracle, __abstractmethods__=set())
def test_query_returns_preferences(segment_samples):
    queries = [segment_samples]

    with patch.object(AbstractOracle, "answer"):
        preference_querent = AbstractOracle()

        preferences = preference_querent.query_preferences(queries)

        assert len(preferences) == len(queries)
        assert type(preferences[0]) is tuple
        assert preferences[0][0] == segment_samples
