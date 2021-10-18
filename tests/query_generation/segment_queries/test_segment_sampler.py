import pytest

from query_generation.segment_queries.segment_sampler import AbstractSegmentSampler
from wrappers.internal.trajectory_buffer import Segment, Buffer


class ConcreteSegmentSampler(AbstractSegmentSampler):
    def _draw_segment_sample(self, trajectory_buffer):
        return Segment(observations=[], actions=[], rewards=[], dones=[], infos=[])


def test_sampler_samples_correct_number_of_samples():
    segment_sampler = ConcreteSegmentSampler(segment_length=0)
    buffer = Buffer(buffer_size=1)
    num_samples = 10

    samples = segment_sampler.sample_segments(num_samples, buffer)

    assert len(samples) == num_samples


def test_sampler_raises_error_when_buffer_has_fewer_elements_than_desired_segment_length():
    # require segment_length of 1 but provide empty buffer
    segment_sampler = ConcreteSegmentSampler(segment_length=1)  # segment_len = 1
    buffer = Buffer(buffer_size=1)  # empty buffer

    with pytest.raises(AssertionError):
        segment_sampler.sample_segments(num_segment_samples=10, trajectory_buffer=buffer)
