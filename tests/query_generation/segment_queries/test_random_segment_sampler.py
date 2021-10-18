import pytest

from query_generation.segment_queries.segment_sampler import RandomSegmentSampler
from wrappers.internal.trajectory_buffer import Buffer, Segment


@pytest.fixture()
def buffer():
    buffer = Buffer(buffer_size=3)
    for _ in range(3):  # fill buffer with dummy data
        obs, act, rew, done, info = 1, 1, 1, 1, {}
        buffer.append_step(obs, act, rew, done, info)

    return buffer


def test_samples_are_segments(buffer):
    segment_sampler = RandomSegmentSampler(segment_length=2)

    samples = segment_sampler.sample_segments(num_segment_samples=1, trajectory_buffer=buffer)

    assert isinstance(samples[0], Segment)


def test_samples_have_correct_length(buffer):
    segment_length = 2
    segment_sampler = RandomSegmentSampler(segment_length)

    samples = segment_sampler.sample_segments(num_segment_samples=1, trajectory_buffer=buffer)

    assert len(samples[0]) == segment_length


def test_segment_sample_is_subsegment_of_buffered_trajectory():
    segment_sampler = RandomSegmentSampler(segment_length=2)
    buffer = Buffer(buffer_size=3)
    for i in range(3):
        obs, act, rew, done, info = i, 1, 1, 1, {}
        buffer.append_step(obs, act, rew, done, info)

    samples = segment_sampler.sample_segments(num_segment_samples=1, trajectory_buffer=buffer)

    def is_subsegment(sample_segment):
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

    assert is_subsegment(samples[0])
