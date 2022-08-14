import os
from typing import List, Tuple

import cv2
import numpy as np
import requests

from .querent import AbstractPreferenceQuerent
from .query_selection.selector import AbstractQuerySelector
from ..environment_wrappers.internal.trajectory_observation.segment import FrameSegment
from ..preference_data.query import Query, BinaryChoiceSetQuery


class HumanPreferenceQuerent(AbstractPreferenceQuerent):

    def __init__(self, query_selector: AbstractQuerySelector, pref_collect_address: str, video_output_directory: str,
                 frames_per_second: int = 20):
        """
        This preference querent queries a human user via a web service. It assumes binary choice set queries with
        trajectory segments as choice alternatives. After selecting the specified number of queries from the set of
        candidates, it puts these selected queries at the query endpoint of the web service, renders videos from the
        trajectory segments of these queries and makes these videos available to the web service by saving these videos
        to the specified location.
        :param query_selector: The query selector that is used for selecting queries from the query candidates.
        :param pref_collect_address: The url of the web service that is used by the user to answer the selected queries.
        :param video_output_directory: The location where the rendered segment videos are saved to.
        :param frames_per_second: The number of frames per second with which the videos are rendered.
        """
        super().__init__(query_selector)
        self.fps = frames_per_second
        self.video_output_dir = self._ensure_dir(video_output_directory)
        self.query_endpoint = pref_collect_address + "/preferences/query/"

    def query_preferences(self, query_candidates: List[BinaryChoiceSetQuery], num_queries: int) -> List[Query]:
        selected_queries = self.query_selector.select_queries(query_candidates, num_queries)

        query: BinaryChoiceSetQuery
        for query in selected_queries:
            self._write_segment_video(query[0], name=f'{query.id}-left')
            self._write_segment_video(query[1], name=f'{query.id}-right')
            requests.put(self.query_endpoint + query.id, json={"uuid": "{}".format(query.id)})

        return selected_queries

    def _write_segment_video(self, segment: FrameSegment, name: str) -> None:

        output_file_name = f'{self.video_output_dir}{name}.webm'
        frame_shape = self._get_frame_shape(segment)

        video_writer = cv2.VideoWriter(output_file_name, cv2.VideoWriter_fourcc(*'VP90'), self.fps, frame_shape)

        for frame in segment.frames:
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        video_writer.release()

    @staticmethod
    def _get_frame_shape(segment: FrameSegment) -> Tuple[int, int]:
        single_frame = np.array(segment.frames[0])
        return single_frame.shape[1], single_frame.shape[0]

    @staticmethod
    def _ensure_dir(base_dir: str):
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        return base_dir
