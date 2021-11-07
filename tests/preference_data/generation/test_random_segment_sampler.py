from collections import deque

import pytest

from query_generation.segment_queries.segment_sampler import RandomSegmentSamplerMixin


@pytest.fixture()
def random_segment_sampler():
    return RandomSegmentSamplerMixin(segment_samples=deque(maxlen=10),
                                     trajectory_buffer=deque(maxlen=10), segment_length=5)


def test_samples_subsegment(random_segment_sampler):
    buffer = random_segment_sampler.trajectory_buffer
    random_segment_sampler.segment_length = 2

    buffer.append(1)
    buffer.append(2)
    buffer.append(3)

    segment = random_segment_sampler.draw_segment_sample()

    def segment_is_subsegment_of_buffered_experiences(sample_segment):
        first_experience = sample_segment[0]
        most_recent_experience = sample_segment[0]
        for current_experience in sample_segment:
            if current_experience != first_experience and current_experience != most_recent_experience + 1:
                return False
            most_recent_experience = current_experience
        return True

    assert segment_is_subsegment_of_buffered_experiences(segment)


def test_sampled_segment_has_correct_length(random_segment_sampler):
    buffer = deque(maxlen=3)
    buffer.append(1)
    buffer.append(2)
    buffer.append(3)

    random_segment_sampler.trajectory_buffer = buffer
    random_segment_sampler.segment_length = 1

    segment_len_1 = random_segment_sampler.draw_segment_sample()

    random_segment_sampler.segment_length = 2
    segment_len_2 = random_segment_sampler.draw_segment_sample()

    random_segment_sampler.segment_length = 0
    segment_len_0 = random_segment_sampler.draw_segment_sample()

    assert len(segment_len_0) == 0
    assert len(segment_len_1) == 1
    assert len(segment_len_2) == 2
