from collections import deque
from unittest.mock import Mock

import pytest

from data_generation.experience import Experience
from data_generation.segment_sampler import TrajectorySegmentSampler


@pytest.fixture()
def segment_sampler():
    return TrajectorySegmentSampler(deque(maxlen=10), segment_length=5)


def test_samples_subsegment(segment_sampler):
    buffer = segment_sampler.trajectory_buffer
    segment_sampler.segment_length = 2

    buffer.append(1)
    buffer.append(2)
    buffer.append(3)

    segment = segment_sampler.generate_sample()

    def segment_is_subsegment_of_buffered_experiences(sample_segment):
        first_experience = sample_segment[0]
        most_recent_experience = sample_segment[0]
        for current_experience in sample_segment:
            if current_experience != first_experience and current_experience != most_recent_experience + 1:
                return False
            most_recent_experience = current_experience
        return True

    assert segment_is_subsegment_of_buffered_experiences(segment)


def test_sampled_segment_has_correct_length(segment_sampler):
    buffer = deque(maxlen=3)
    buffer.append(1)
    buffer.append(2)
    buffer.append(3)

    segment_sampler.trajectory_buffer = buffer
    segment_sampler.segment_length = 1

    segment_len_1 = segment_sampler.generate_sample()

    segment_sampler.segment_length = 2
    segment_len_2 = segment_sampler.generate_sample()

    segment_sampler.segment_length = 0
    segment_len_0 = segment_sampler.generate_sample()

    assert len(segment_len_0) == 0
    assert len(segment_len_1) == 1
    assert len(segment_len_2) == 2


def test_saves_sampled_trajectory_segment(segment_sampler):
    trajectory_segment = [Experience(1), Experience(2)]

    segment_sampler.generate_sample = Mock(return_value=trajectory_segment)

    segment_sampler.save_sample()

    assert trajectory_segment in segment_sampler.segment_samples
