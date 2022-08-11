import os
from typing import List

import cv2
import numpy as np
import requests

from pref_rl.preference_querying.querent import AbstractPreferenceQuerent


class HumanPreferenceQuerent(AbstractPreferenceQuerent):

    def __init__(self, query_selector, pref_collect_address, video_output_directory, frames_per_second=20):
        super().__init__(query_selector)
        self.fps = frames_per_second
        self.video_output_dir = self._ensure_dir(video_output_directory)
        self.query_endpoint = pref_collect_address + "/preferences/query/"

    def query_preferences(self, query_candidates, num_queries) -> List:
        selected_queries = self.query_selector.select_queries(query_candidates, num_queries)

        for query in selected_queries:
            self._write_segment_video(query[0], name=f'{query.id}-left')
            self._write_segment_video(query[1], name=f'{query.id}-right')
            requests.put(self.query_endpoint + query.id, json={"uuid": "{}".format(query.id)})

        return selected_queries

    def _write_segment_video(self, segment, name, fourcc=cv2.VideoWriter_fourcc(*'VP90'), file_extension='.webm'):

        output_file_name = f'{self.video_output_dir}{name}{file_extension}'
        frame_shape = self._get_frame_shape(segment)

        video_writer = cv2.VideoWriter(output_file_name, fourcc, self.fps, frame_shape)

        for frame in segment.frames:
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        video_writer.release()

    @staticmethod
    def _get_frame_shape(segment):
        single_frame = np.array(segment.frames[0])
        return single_frame.shape[1], single_frame.shape[0]

    @staticmethod
    def _ensure_dir(base_dir):
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        return base_dir
