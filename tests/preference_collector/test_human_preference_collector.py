import pytest
import django
import os
import sys
from preference_collector.binary_choice import BinaryChoice
from preference_collector.human_preference.human_preference_collector import HumanPreferenceCollector
from query_generator.query import Query


def test_human_pref_collector(preference_collector):
    just_collected_preferences = preference_collector.collect_preferences()
    assert len(just_collected_preferences) == 3
    assert just_collected_preferences[0].choice == BinaryChoice.LEFT
    assert just_collected_preferences[1].choice == BinaryChoice.INDIFFERENT
    assert just_collected_preferences[2].choice == BinaryChoice.RIGHT


@pytest.fixture()
def preference_collector():
    sys.path.append('./preference_collection_webapp/')
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pbrlwebapp.settings')
    django.setup()

    from preferences import models

    pending_queries = [Query() for _ in range(4)]

    i = 1
    for query in pending_queries:
        pref = models.Preference(uuid=query.id)
        pref.label = i
        pref.full_clean()
        pref.save()
        i -= .5

    preference_collector = HumanPreferenceCollector()
    preference_collector.pending_queries = pending_queries

    yield preference_collector

    prefs_to_delete = models.Preference.objects.order_by('-id')[:4]
    for pref in prefs_to_delete:
        pref.delete()
