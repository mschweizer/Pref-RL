from _typeshed import FileDescriptor
import os
import pytest
from unittest.mock import Mock, MagicMock

from preference_querent.human_preference.human_preference_querent import HumanPreferenceQuerent
from preference_querent.query_selector.query_selector import RandomQuerySelector
from query_generator.query import BinaryChoiceQuery

def test_human_pref_querent(video_directory):

    human_preference_querent = HumanPreferenceQuerent(
        query_selector=RandomQuerySelector(), video_root_output_dir=video_directory)

    segment1 = Mock()
    segment1.frames = MagicMock(return_value=[[[[255, 255, 255]]]])
    segment2 = Mock()
    segment2.frames = MagicMock(return_value=[[[[255, 255, 255]]]])

    test_query = BinaryChoiceQuery([segment1, segment2])

    human_preference_querent.query_preferences(
        query_candidates=[test_query], num_queries=1)

    dir_count = 0
    file_count = 0
    for _, dirs, files in os.walk(video_directory):
        for _ in dirs:
            dir_count += 1
        for _ in files:
            file_count += 1
    assert dir_count == 1
    assert file_count == 2


@pytest.fixture()
def prepare_video_directory():
    video_directory = '/tmp/videofiles/'
    if not os.path.exists(video_directory):
        os.makedirs(video_directory)

    yield video_directory

    for _, _, files in os.walk(video_directory):
        for file in files:
            os.remove(file)
    os.rmdir(video_directory)