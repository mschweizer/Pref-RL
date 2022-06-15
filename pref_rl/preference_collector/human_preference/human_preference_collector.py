from typing import List

import requests

from pref_rl.preference_collector.binary_choice import BinaryChoice
from pref_rl.preference_collector.preference import Preference
from pref_rl.preference_collector.preference_collector import AbstractPreferenceCollector


INCOMPARABLE = -1.
ERROR_MSG = "Unexpected value for label retrieved from database. "


class HumanPreferenceCollector(AbstractPreferenceCollector):

    def __init__(self, pref_collect_address):
        super().__init__()
        self.query_endpoint = pref_collect_address + "/preferences/query/"

    def collect_preferences(self) -> List:

        just_collected_preferences = []

        for query in list(self.pending_queries):

            retrieved_label = self._retrieve_label(query.id)

            if retrieved_label is not None:
                preference = self._create_preference(query, retrieved_label)
                just_collected_preferences.extend([preference])
                self.pending_queries.remove(query)

        return just_collected_preferences

    def _create_preference(self, query, retrieved_label):
        choice = self._create_choice(retrieved_label)
        if choice:
            return Preference(query, choice)

    @staticmethod
    def _create_choice(retrieved_label):
        try:
            return BinaryChoice(retrieved_label)
        except ValueError as e:
            if retrieved_label != INCOMPARABLE:
                raise ValueError(ERROR_MSG + str(e))

    def _retrieve_label(self, query_id):
        answered_query = requests.get(self.query_endpoint + query_id).json()
        return answered_query["label"]
