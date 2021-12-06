import mock
from mock import patch
from preference_collector.human_preference.human_preference_collector import HumanPreferenceCollector
from query_generator.query import Query


def test_human_pref_collector(Preference):

    pending_queries = [Query() for _ in range(4)]

    i = -.5

    for query in pending_queries:
        pref = Preference(uuid=query.id)
        pref.label = i
        pref.full_clean()
        pref.save()
        i += .5
