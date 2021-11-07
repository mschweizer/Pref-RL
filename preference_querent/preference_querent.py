from abc import ABC, abstractmethod
from typing import List
import os
import sys
import django
import cv2
import numpy as np


class AbstractPreferenceQuerent(ABC):
    def __init__(self, query_selector):
        self.query_selector = query_selector

    @abstractmethod
    def query_preferences(self, query_candidates, num_queries) -> List:
        pass


class SynchronousPreferenceQuerent(AbstractPreferenceQuerent):

    def __init__(self, query_selector, preference_collector, preferences):
        super(SynchronousPreferenceQuerent, self).__init__(query_selector)
        self.preference_collector = preference_collector
        self.preferences = preferences

    def query_preferences(self, query_candidates, num_queries) -> List:
        newly_pending_queries = self.query_selector.select_queries(
            query_candidates, num_queries)
        self.preference_collector.pending_queries.extend(newly_pending_queries)
        just_collected_preferences = self.preference_collector.collect_preferences()
        self.preferences.extend(just_collected_preferences)
        return []


class DjangoPreferenceQuerent(AbstractPreferenceQuerent):

    def __init__(self, query_selector, base_output_dir):
        super().__init__(query_selector)
        self.base_output_dir = base_output_dir
        if not os.path.exists(base_output_dir):
            os.makedirs(base_output_dir)
        # preparations for django connection
        sys.path.append('/home/sascha/BA/webapp/pref-rl-webapp')
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pbrlwebapp.settings')
        django.setup()

    def query_preferences(self, query_candidates, num_queries) -> List:
        selected_queries = self.query_selector.select_queries(
            query_candidates, num_queries)

        for query in selected_queries:

            from preferences import models

            self._render_segment(
                query[0], subdir='{}/'.format(query.id), name='{}-left'.format(query.id))
            self._render_segment(
                query[1], subdir='{}/'.format(query.id), name='{}-right'.format(query.id))

            models.Preference.objects.create(uuid=query.id)

        return selected_queries

    def _render_segment(self, segment, subdir, name, fps=12, fourcc=cv2.VideoWriter_fourcc(*'vp80'), file_extension='.webm'):
        self._setup_subdir(self.base_output_dir, subdir)
        outfile = '{}{}{}{}'.format(
            self.base_output_dir, subdir, name, file_extension)
        singleframe = np.array(segment.frames[0])
        fshape = (singleframe.shape[1], singleframe.shape[0])

        vid_writer = cv2.VideoWriter(outfile, fourcc, fps, fshape)

        for frame in segment.frames:
            vid_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        vid_writer.release()
        return '{}{}'.format(name, file_extension)

    def _setup_subdir(self, base_dir, subdir_name):
        if not os.path.exists(base_dir+subdir_name):
            os.mkdir(base_dir+subdir_name)
        return subdir_name
