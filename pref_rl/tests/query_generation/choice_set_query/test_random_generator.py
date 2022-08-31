from unittest.mock import Mock

import pytest

from pref_rl.query_generation.choice_set_query.alternative_generation.segment_alternative.trajectory_segment import TrajectorySegment
from ....query_generation.choice_set_query.random_generator import RandomChoiceSetQueryGenerator


@pytest.fixture()
def segment_samples():
    return [TrajectorySegment(observations=[1], actions=[1], rewards=[1], dones=[1], infos=[1]),
            TrajectorySegment(observations=[2], actions=[2], rewards=[2], dones=[2], infos=[2]),
            TrajectorySegment(observations=[3], actions=[3], rewards=[3], dones=[3], infos=[3])]


def test_selected_segments_are_from_segment_samples(segment_samples):
    generator = RandomChoiceSetQueryGenerator(alternative_generator=Mock())

    choice_sets = generator._select_choice_sets(num_choice_sets=1, alternatives=segment_samples)

    for segment in choice_sets[0]:
        assert segment in segment_samples


def test_selects_right_number_of_segments(segment_samples):
    generator = RandomChoiceSetQueryGenerator(alternative_generator=Mock())
    num_segments = 2

    choice_sets = generator._select_choice_sets(num_choice_sets=1, alternatives=segment_samples)

    assert len(choice_sets[0]) == num_segments
