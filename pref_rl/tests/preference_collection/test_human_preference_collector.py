import pytest

from pref_rl.preference_data.binary_choice import BinaryChoice
from ...preference_collection.human.collector import HumanPreferenceCollector, ERROR_MSG, \
    INCOMPARABLE, INCOMPARABLE_ERROR_MSG
from pref_rl.preference_data.preference import Preference
from pref_rl.preference_data.query import BinaryChoiceSetQuery


ADDRESS = "http://url"


@pytest.fixture()
def collector():
    return HumanPreferenceCollector(pref_collect_address=ADDRESS)


@pytest.fixture()
def query():
    return BinaryChoiceSetQuery(["segment_1", "segment_2"])


def test_retrieves_label_from_endpoint(collector, query, requests_mock):
    preference = {"query_id": query.id, "label": 1.0}

    requests_mock.get(ADDRESS + "/preferences/query/{}".format(query.id), json=preference)
    retrieved_label = collector._retrieve_label(query.id)

    assert retrieved_label == preference["label"]


def test_creates_preference_if_value_is_valid(collector):
    for e in BinaryChoice:
        label = e.value
        preference = collector._create_preference(query={}, retrieved_label=label)
        assert BinaryChoice(label) == preference.choice


def test_raises_value_error_when_creating_preference_if_value_is_invalid(collector):
    invalid_label = 100.
    with pytest.raises(ValueError) as e:
        collector._create_preference(query={}, retrieved_label=invalid_label)
    assert ERROR_MSG in e.value.args[0]


def test_raises_value_error_if_query_is_incomparable(collector, query):
    with pytest.raises(ValueError) as e:
        collector._create_preference(query, INCOMPARABLE)
    assert INCOMPARABLE_ERROR_MSG in e.value.args[0]


def test_collects_preference(collector, query, requests_mock):
    collector.pending_queries.append(query)
    label = 1.0
    requests_mock.get(ADDRESS + "/preferences/query/{}".format(query.id), json={"query_id": query.id, "label": label})

    assert Preference(query, BinaryChoice(label)) in collector.collect_preferences()


def test_does_not_collect_preference_if_query_was_incomparable(collector, query, requests_mock):
    collector.pending_queries.append(query)
    label = -1.0
    requests_mock.get(ADDRESS + "/preferences/query/{}".format(query.id), json={"query_id": query.id, "label": label})

    assert len(collector.collect_preferences()) == 0


def test_does_log_value_error_if_query_was_incomparable(collector, query, requests_mock, caplog):
    collector.pending_queries.append(query)
    label = -1.0
    requests_mock.get(ADDRESS + "/preferences/query/{}".format(query.id), json={"query_id": query.id, "label": label})
    collector.collect_preferences()

    assert caplog.records[0].levelname == "DEBUG"
    assert INCOMPARABLE_ERROR_MSG in caplog.records[0].message


def test_collected_query_is_removed_from_pending(collector, query, requests_mock):
    collector.pending_queries.append(query)
    label = 1.0
    requests_mock.get(ADDRESS + "/preferences/query/{}".format(query.id), json={"query_id": query.id, "label": label})
    collector.collect_preferences()

    assert query not in collector.pending_queries


def test_not_collected_query_is_not_removed_from_pending(collector, query, requests_mock):
    collector.pending_queries.append(query)
    requests_mock.get(ADDRESS + "/preferences/query/{}".format(query.id), json={"query_id": query.id, "label": None})
    collector.collect_preferences()

    assert query in collector.pending_queries


def test_incomparable_query_is_removed_from_pending(collector, query, requests_mock):
    collector.pending_queries.append(query)
    requests_mock.get(ADDRESS + "/preferences/query/{}".format(query.id),
                      json={"query_id": query.id, "label": INCOMPARABLE})
    collector.collect_preferences()

    assert query not in collector.pending_queries
