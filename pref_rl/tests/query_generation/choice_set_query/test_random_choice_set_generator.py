from unittest.mock import Mock

import pytest

from ....environment_wrappers.internal.trajectory_observation.segment import Segment
from ....query_generation.choice_set_query.generator import RandomChoiceSetQueryGenerator


@pytest.fixture()
def segment_samples():
    return [Segment(observations=[1], actions=[1], rewards=[1], dones=[1], infos=[1]),
            Segment(observations=[2], actions=[2], rewards=[2], dones=[2], infos=[2]),
            Segment(observations=[3], actions=[3], rewards=[3], dones=[3], infos=[3])]


def test_selected_segments_are_from_segment_samples(segment_samples):
    generator = RandomChoiceSetQueryGenerator(alternative_generator=Mock())

    choice_sets = generator.select_choice_sets(num_choice_sets=1, num_alternatives_per_choice_set=2,
                                               alternatives=segment_samples)

    for segment in choice_sets[0]:
        assert segment in segment_samples


def test_selects_right_number_of_segments(segment_samples):
    generator = RandomChoiceSetQueryGenerator(alternative_generator=Mock())
    num_segments = 2

    choice_sets = generator.select_choice_sets(num_choice_sets=1, num_alternatives_per_choice_set=num_segments,
                                               alternatives=segment_samples)

    assert len(choice_sets[0]) == num_segments
