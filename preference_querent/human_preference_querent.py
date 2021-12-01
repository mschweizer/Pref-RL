from preference_querent.preference_querent import AbstractPreferenceQuerent
from typing import List
import os
import sys
import django
import cv2
import numpy as np


class HumanPreferenceQuerent(AbstractPreferenceQuerent):

    def __init__(self, query_selector, video_root_output_dir='./videofiles/'):
        super().__init__(query_selector)
        self.root_output_dir = video_root_output_dir
        if not os.path.exists(video_root_output_dir):
            os.makedirs(video_root_output_dir)
        # preparations for django connection
        sys.path.append(os.path.abspath('./preference_collection_webapp'))
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pbrlwebapp.settings')
        django.setup()

    def query_preferences(self, query_candidates, num_queries) -> List:
        selected_queries = self.query_selector.select_queries(
            query_candidates, num_queries)

        for query in selected_queries:

            self._write_segment_video(
                query[0], subdir=f'{query.id}/', name=f'{query.id}-left')
            self._write_segment_video(
                query[1], subdir=f'{query.id}/', name=f'{query.id}-right')

            from preferences import models
            models.Preference.objects.create(uuid=query.id)

        return selected_queries

    def _write_segment_video(self, segment, subdir, name, fps=8, fourcc=cv2.VideoWriter_fourcc(*'VP80'), file_extension='.webm'):

        self._ensure_subdir(self.root_output_dir, subdir)
        output_file = f'{self.root_output_dir}{subdir}{name}{file_extension}'
        frame_shape = self._get_frame_shape(segment)

        video_writer = cv2.VideoWriter(output_file, fourcc, fps, frame_shape)

        for frame in segment.frames:
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        video_writer.release()

    def _ensure_subdir(self, base_dir, subdir):
        if not os.path.exists(f'{base_dir}{subdir}'):
            os.mkdir(f'{base_dir}{subdir}')

    def _get_frame_shape(self, segment):
        single_frame = np.array(segment.frames[0])
        return (single_frame.shape[1], single_frame.shape[0])
