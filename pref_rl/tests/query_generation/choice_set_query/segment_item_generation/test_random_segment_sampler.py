from unittest.mock import Mock

import pytest

from .....agents.policy.buffered_model import BufferedPolicyModel
from .....environment_wrappers.internal.trajectory_observation.buffer import Buffer
from .....environment_wrappers.internal.trajectory_observation.segment import Segment
from .....query_generation.choice_set_query.item_generation.segment_item.random_sampler import RandomSegmentSampler


@pytest.fixture()
def policy_model():
    buffer = Buffer(buffer_size=3)
    for _ in range(3):  # fill buffer with dummy data
        obs, act, rew, done, info = 1, 1, 1, 1, {}
        buffer.append_step(obs, act, rew, done, info)

    policy_model = Mock(spec_set=BufferedPolicyModel)
    policy_model.trajectory_buffer = buffer

    return policy_model


def test_samples_are_segments(policy_model):
    segment_sampler = RandomSegmentSampler(segment_length=2)

    samples = segment_sampler.generate(policy_model, num_items=1)

    assert isinstance(samples[0], Segment)


def test_samples_have_correct_length(policy_model):
    segment_length = 2
    segment_sampler = RandomSegmentSampler(segment_length)

    samples = segment_sampler.generate(policy_model, num_items=1)

    assert len(samples[0]) == segment_length


def test_segment_sample_is_subsegment_of_buffered_trajectory():
    buffer = Buffer(buffer_size=3)
    for i in range(3):
        obs, act, rew, done, info = i, 1, 1, 1, {}
        buffer.append_step(obs, act, rew, done, info)

    policy_model = Mock(spec_set=BufferedPolicyModel)
    policy_model.trajectory_buffer = buffer

    segment_sampler = RandomSegmentSampler(segment_length=2)

    samples = segment_sampler.generate(policy_model, num_items=1)

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
