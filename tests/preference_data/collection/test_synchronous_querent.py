from unittest.mock import patch

from preference_data.querent.synchronous.synchronous_querent import AbstractSynchronousPreferenceQuerent


@patch.multiple(AbstractSynchronousPreferenceQuerent, __abstractmethods__=set())
def test_query_returns_preferences(segment_samples):
    queries = [segment_samples]

    with patch.object(AbstractSynchronousPreferenceQuerent, "answer"):
        preference_querent = AbstractSynchronousPreferenceQuerent()

        preferences = preference_querent.query_preferences(queries)

        assert len(preferences) == len(queries)
        assert type(preferences[0]) is tuple
        assert preferences[0][0] == segment_samples
