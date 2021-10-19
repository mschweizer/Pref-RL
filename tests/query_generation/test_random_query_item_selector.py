import pytest

from query_generation.query_item_selector import RandomQueryItemSelector
from wrappers.internal.trajectory_buffer import Segment


@pytest.fixture()
def segment_samples():
    return [Segment(observations=[1], actions=[1], rewards=[1], dones=[1], infos=[1]),
            Segment(observations=[2], actions=[2], rewards=[2], dones=[2], infos=[2]),
            Segment(observations=[3], actions=[3], rewards=[3], dones=[3], infos=[3])]


def test_selected_segments_are_from_segment_samples(segment_samples):
    segment_selector = RandomQueryItemSelector()

    selected_segments = segment_selector.select_items(segment_samples, num_items=2)

    for segment in selected_segments:
        assert segment in segment_samples


def test_selects_right_number_of_segments(segment_samples):
    segment_selector = RandomQueryItemSelector()
    num_segments = 2

    selected_segments = segment_selector.select_items(segment_samples, num_segments)

    assert len(selected_segments) == num_segments
