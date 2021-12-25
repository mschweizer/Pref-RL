from typing import List
from preference_collector.preference import Preference
from preference_collector.preference_collector import AbstractPreferenceCollector
from preference_collector.binary_choice import BinaryChoice


class HumanPreferenceCollector(AbstractPreferenceCollector):

    def __init__(self):
        super().__init__()

    def collect_preferences(self) -> List:
        from preferences import models

        just_collected_preferences = []

        for query in list(self.pending_queries):
            db_pref = models.Preference.objects.get(uuid=str(query.id))

            if (retrieved_label := db_pref.label) is None:
                continue

            pref_rl_label: BinaryChoice

            if retrieved_label == 1:
                pref_rl_label = BinaryChoice.LEFT
            elif retrieved_label == .5:
                pref_rl_label = BinaryChoice.INDIFFERENT
            elif retrieved_label == 0:
                pref_rl_label = BinaryChoice.RIGHT
            elif retrieved_label < 0:  # query could not be answered
                self.pending_queries.remove(query)
                continue
            else:
                raise ValueError(
                    'Unexpected value for label retrieved from database.')

            just_collected_preferences.append(
                Preference(query=query, choice=pref_rl_label))
            self.pending_queries.remove(query)

        return just_collected_preferences