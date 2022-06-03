import os
from typing import List

import cv2
import numpy as np
import requests

from ..preference_querent import AbstractPreferenceQuerent


def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def _ensure_subdir(base_dir, subdir):
    if not os.path.exists(f'{base_dir}{subdir}'):
        os.mkdir(f'{base_dir}{subdir}')


def _get_frame_shape(segment):
    single_frame = np.array(segment.frames[0])
    return single_frame.shape[1], single_frame.shape[0]


class HumanPreferenceQuerent(AbstractPreferenceQuerent):

    # TODO: remove output_dir / propagate param to agent config
    def __init__(self, query_selector, video_root_output_dir='/home/yp5266/PycharmProjects/pref_collect/videofiles/'):
        super().__init__(query_selector)
        self.root_output_dir = ensure_dir(video_root_output_dir)

    def query_preferences(self, query_candidates, num_queries) -> List:
        selected_queries = self.query_selector.select_queries(
            query_candidates, num_queries)

        for query in selected_queries:
            self._write_segment_video(
                query[0], subdir=f'', name=f'{query.id}-left')
            self._write_segment_video(
                query[1], subdir=f'', name=f'{query.id}-right')

            # TODO: make address configurable
            response = requests.put('http://127.0.0.1:8000/preferences/query/{}'.format(query.id),
                                    json={"uuid": "{}".format(query.id)})

        return selected_queries

    def _write_segment_video(self, segment, subdir, name, fps=20, fourcc=cv2.VideoWriter_fourcc(*'VP90'),
                             file_extension='.webm'):

        _ensure_subdir(self.root_output_dir, subdir)
        output_file_name = f'{self.root_output_dir}{subdir}{name}{file_extension}'
        frame_shape = _get_frame_shape(segment)

        video_writer = cv2.VideoWriter(output_file_name, fourcc, fps, frame_shape)

        for frame in segment.frames:
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        video_writer.release()
