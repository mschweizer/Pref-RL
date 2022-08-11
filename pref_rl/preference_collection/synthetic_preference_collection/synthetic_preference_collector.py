from typing import List

from pref_rl.preference_data.preference import Preference
from ..preference_collector import AbstractPreferenceCollector


class SyntheticPreferenceCollector(AbstractPreferenceCollector):

    def __init__(self, oracle):
        super(SyntheticPreferenceCollector, self).__init__()
        self.oracle = oracle

    def collect_preferences(self) -> List:
        just_collected_preferences = [Preference(query=query, choice=self.oracle.answer(query))
                                      for query in self.pending_queries]
        self.pending_queries.clear()
        return just_collected_preferences
