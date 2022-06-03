from typing import List

import requests

from pref_rl.preference_collector.binary_choice import BinaryChoice
from pref_rl.preference_collector.preference import Preference
from pref_rl.preference_collector.preference_collector import AbstractPreferenceCollector


class HumanPreferenceCollector(AbstractPreferenceCollector):

    def __init__(self):
        super().__init__()

    def collect_preferences(self) -> List:

        just_collected_preferences = []

        for query in list(self.pending_queries):
            # TODO: make address configurable
            response = requests.get('http://127.0.0.1:8000/preferences/query/{}'.format(query.id))
            answered_query = response.json()

            if (retrieved_label := answered_query["label"]) is None:
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
