from typing import List
from preference_collector.preference import Preference
from preference_collector.preference_collector import AbstractPreferenceCollector
from preference_collector.binary_choice import BinaryChoice
import os
import sys
import django


class DjangoPreferenceCollector(AbstractPreferenceCollector):

    def __init__(self):
        super().__init__()
        sys.path.append('/home/sascha/BA/webapp/pref-rl-webapp')
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pbrlwebapp.settings')
        django.setup()

    def collect_preferences(self) -> List:
        from preferences import models

        just_collected_preferences = []

        for query in self.pending_queries:
            db_pref = models.Preference.objects.get(uuid=str(query.id))

            if (retrieved_label := db_pref.label) is None:
                continue

            label = None

            if retrieved_label == 1:
                label = BinaryChoice.LEFT
            elif retrieved_label == .5:
                label = BinaryChoice.INDIFFERENT
            elif retrieved_label == 0:
                label = BinaryChoice.RIGHT
            just_collected_preferences.append(
                Preference(query=query, choice=label))
            self.pending_queries.remove(query)

        return just_collected_preferences
