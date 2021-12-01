import django
from django.conf import settings
from preference_collection_webapp.pbrlwebapp import settings as webapp_settings
from preference_collector.human_preference.human_preference_collector import HumanPreferenceCollector
from query_generator.query import Query

def test_human_pref_collector():

    settings.configure()
    django.setup(webapp_settings)
    from preference_collection_webapp.preferences import models
    
    preference_collector = HumanPreferenceCollector()

    pending_queries = [Query() for _ in range(4)]

    i = -.5
    
    for query in pending_queries:
        pref = models.Preference(uuid=query.id)
        pref.label = i
        pref.full_clean()
        pref.save()
        i += .5
    
    preference_collector.pending_queries = pending_queries.copy()

    preference_collector.collect_preferences()

    assert len(preference_collector.pending_queries) == 3
    for query in preference_collector.pending_queries:
        assert query.label >= 0