from unittest.mock import Mock

import numpy as np
import pytest

from pref_rl.preference_querying.human_preference_querent import HumanPreferenceQuerent
from pref_rl.preference_querying.query_selection.query_selector import RandomQuerySelector
from pref_rl.query_generation.query import ChoiceQuery

ADDRESS = "http://url"


@pytest.fixture()
def segment():
    frame = np.random.randint(0, high=255, size=(2, 2, 3), dtype=np.uint8)
    segment = Mock()
    segment.frames = np.array([frame, frame, frame])
    return segment


@pytest.fixture()
def query(segment):
    return ChoiceQuery(choice_set=[segment, segment])


@pytest.fixture()
def querent(tmpdir):
    return HumanPreferenceQuerent(query_selector=RandomQuerySelector(), pref_collect_address=ADDRESS,
                                  video_output_directory=str(tmpdir) + "/")


def test_writes_video_file(querent, segment, tmpdir):
    filename = "video"
    querent._write_segment_video(segment, name=filename)

    assert tmpdir.join("/{}.webm".format(filename)).exists()


def test_queries_correct_number_of_queries(querent, query, tmpdir, requests_mock):
    requests_mock.put(ADDRESS + "/preferences/query/{}".format(query.id))

    num_queries = 1
    newly_pending_queries = querent.query_preferences(query_candidates=[query], num_queries=num_queries)

    assert len(newly_pending_queries) == num_queries


def test_writes_videos_for_queries(querent, query, tmpdir, requests_mock):
    requests_mock.put(ADDRESS + "/preferences/query/{}".format(query.id))
    querent.query_preferences(query_candidates=[query], num_queries=1)

    assert tmpdir.join("/{}-left.webm".format(query.id)).exists()
    assert tmpdir.join("/{}-right.webm".format(query.id)).exists()


def test_identifies_correct_frame_shape(querent):
    frame_shape = [2, 2]
    frame = np.random.randint(0, high=255, size=frame_shape + [3], dtype=np.uint8)
    segment = Mock()
    segment.frames = np.array([frame, frame, frame])

    assert querent._get_frame_shape(segment) == tuple(frame_shape)


def test_creates_video_dir_if_not_existent(querent, tmpdir):
    does_not_exist_dir = str(tmpdir) + "/does_not_exist/"
    assert not tmpdir.join("/does_not_exist/").exists()
    querent._ensure_dir(does_not_exist_dir)
    assert tmpdir.join("/does_not_exist/").exists()
