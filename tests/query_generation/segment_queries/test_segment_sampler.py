from unittest.mock import Mock

import pytest

from agents.policy_model import BufferedPolicyModel
from query_generation.segment_queries.segment_sampler import AbstractSegmentSampler
from wrappers.internal.trajectory_buffer import Buffer


class ConcreteSegmentSampler(AbstractSegmentSampler):
    def _sample_segment(self, trajectory_buffer):
        return "sample"


def test_sampler_samples_correct_number_of_samples():
    policy_model = Mock(spec_set=BufferedPolicyModel)
    policy_model.trajectory_buffer = Buffer(buffer_size=1)

    segment_sampler = ConcreteSegmentSampler(segment_length=0)
    num_samples = 10

    samples = segment_sampler.generate(policy_model, num_samples)

    assert len(samples) == num_samples


def test_sampler_raises_error_when_buffer_has_fewer_elements_than_desired_segment_length():
    policy_model = Mock(spec_set=BufferedPolicyModel)

    # require segment_length of 1 but provide empty buffer
    policy_model.trajectory_buffer = Buffer(buffer_size=1)  # empty buffer
    segment_sampler = ConcreteSegmentSampler(segment_length=1)  # segment_len = 1

    with pytest.raises(AssertionError):
        segment_sampler.generate(policy_model, num_items=10)
