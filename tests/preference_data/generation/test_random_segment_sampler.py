from collections import deque

import pytest

from query_generation.segment_queries.segment_sampler import RandomSegmentSamplerMixin
from wrappers.internal.trajectory_buffer import Buffer


@pytest.fixture()
def random_segment_sampler():
    return RandomSegmentSamplerMixin(segment_samples=deque(maxlen=10),
                                     trajectory_buffer=Buffer(buffer_size=10), segment_length=5)


def test_samples_subsegment(random_segment_sampler):
    buffer = random_segment_sampler.trajectory_buffer
    random_segment_sampler.segment_length = 2

    buffer.append_step(1, "act", "rew", "done", "info")
    buffer.append_step(2, "act", "rew", "done", "info")
    buffer.append_step(3, "act", "rew", "done", "info")

    segment = random_segment_sampler.draw_segment_sample()

    def segment_is_subsegment_of_buffered_experiences(sample_segment):
        first_experience = sample_segment.get_step(0)
        most_recent_experience = sample_segment.get_step(0)
        for i in range(len(sample_segment)):
            current_experience = sample_segment.get_step(i)
            if current_experience["observation"] \
                    != first_experience["observation"] and current_experience["observation"] \
                    != most_recent_experience["observation"] + 1:
                return False
            most_recent_experience = current_experience
        return True

    assert segment_is_subsegment_of_buffered_experiences(segment)


def test_sampled_segment_has_correct_length(random_segment_sampler):
    buffer = Buffer(buffer_size=10)
    buffer.append_step("obs", "act", "rew", "done", "info")
    buffer.append_step("obs", "act", "rew", "done", "info")
    buffer.append_step("obs", "act", "rew", "done", "info")

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
