from abc import ABC, abstractmethod
from uuid import uuid4

from preference_collection.preference_oracle import RewardMaximizingOracleMixin
from query_selection.query_selector import AbstractQuerySelectorMixin, MostRecentlyGeneratedQuerySelectorMixin
from video.segment_renderer import SegmentRenderer


class AbstractPreferenceCollectorMixin(AbstractQuerySelectorMixin, ABC):

    def __init__(self, preferences, query_candidates):
        self.query_candidates = query_candidates
        self.preferences = preferences

    @abstractmethod
    def query_preferences(self, num_preferences):
        pass


class BaseSyntheticPreferenceCollectorMixin(AbstractPreferenceCollectorMixin,
                                            MostRecentlyGeneratedQuerySelectorMixin, RewardMaximizingOracleMixin):

    def query_preferences(self, num_preferences):
        queries = self.select_queries(
            self.query_candidates, num_queries=num_preferences)
        self.preferences.extend([(query, self.answer(query))
                                for query in queries])


class BaseHumanPreferenceCollectorMixin(AbstractPreferenceCollectorMixin, MostRecentlyGeneratedQuerySelectorMixin):

    def __init__(self, preferences, query_candidates, output_path='./videofiles/'):
        super().__init__(preferences, query_candidates)
        self.renderer = SegmentRenderer(output_path=output_path)

    def query_preferences(self, num_preferences):
        queries = self.select_queries(
            self.query_candidates, num_queries=num_preferences)
        for query in queries:
            uuid = uuid4()
            str_id = str(uuid)
            self.renderer.render_segment(query[0], '{}-left'.format(str_id))
            self.renderer.render_segment(query[1], '{}-right'.format(str_id))
            # add query to db, link to saved clips

    def collect_preferences():
        # TODO implement collection
        pass
