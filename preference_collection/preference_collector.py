from abc import ABC, abstractmethod
from uuid import uuid4
from sys import path
import os
import django
from preference_collection.label import Label
from math import trunc


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
        self._queried_prefs = []

        # preparations for django connection
        path.append('/home/sascha/BA/webapp/pref-rl-webapp')
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pbrlwebapp.settings')
        django.setup()

    def query_preferences(self, num_preferences):
        queries = self.select_queries(
            self.query_candidates, num_queries=num_preferences)
        for query in queries:
            uuid = uuid4()
            str_id = str(uuid)
            url_left = self.renderer.render_segment(
                query[0], '{}-left'.format(str_id))
            url_right = self.renderer.render_segment(
                query[1], '{}-right'.format(str_id))

            from preferences import models
            unlabeled_preference = models.Preference.objects.create(
                video_url_left=url_left, video_url_right=url_right)
            unlabeled_preference.save()
            self._queried_prefs.append(
                {'id': unlabeled_preference.id, 'query': query, 'label': None})
            pass

    def collect_preferences(self):
        from preferences import models
        for pref in self._unlabeled_prefs:
            db_pref = models.Preference.objects.get(pk=pref['id'])
            retrieved_label = db_pref.label
            if retrieved_label is not None:
                pref['label'] = retrieved_label
                self.preferences.append(
                    (pref['query'], self._convert_label(retrieved_label)))
    
    def clear_queried_prefs(self):
        self._queried_prefs = []

    @property
    def label_quota(self):
        if not len(self._queried_prefs)>0:
            return 0.0
        return trunc((len(self._labeled_prefs)/len(self._queried_prefs))*100)/100

    @property
    def _unlabeled_prefs(self):
        return [pref for pref in self._queried_prefs if pref['label'] is None]

    @property
    def _labeled_prefs(self):
        return [pref for pref in self._queried_prefs if pref['label'] is not None]

    def _convert_label(self, integer_label):
        if integer_label < 0:
            return Label.LEFT
        elif integer_label > 0:
            return Label.RIGHT
        else:
            return Label.INDIFFERENT
