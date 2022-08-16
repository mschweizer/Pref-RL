import logging
from unittest.mock import MagicMock, Mock

import pytest

from .....agents.policy.buffered_model import ObservedPolicyModel
from .....environment_wrappers.internal.trajectory_observation.buffer import Buffer
from .....environment_wrappers.internal.trajectory_observation.observer import TrajectoryObserver
from .....environment_wrappers.internal.trajectory_observation.segment import Segment
from .....query_generation.choice_set_query.alternative_generation.segment_alternative.sampler import SegmentSampler


class ConcreteSegmentSampler(SegmentSampler):
    def __init__(self, segment_length):
        super().__init__(segment_length)
        self.logger = logging.getLogger()

    def _sample_segment(self, trajectory_buffer):
        return "segment"


@pytest.fixture()
def policy_model():
    buffer = Buffer(buffer_size=3)
    for _ in range(3):  # fill buffer with dummy data
        obs, act, rew, done, info = 1, 1, 1, 1, {}
        buffer.append_step(obs, act, rew, done, info)

    policy_model = Mock()
    policy_model.trajectory_buffer = buffer

    return policy_model


@pytest.fixture()
def segment_sampler():
    return SegmentSampler(segment_length=1)


def test_samples_correct_number_of_segments(cartpole_env):
    segment_sampler = SegmentSampler(segment_length=1)
    num_segments = 10
    policy_model = ObservedPolicyModel(TrajectoryObserver(cartpole_env, trajectory_buffer_size=1000), train_freq=5)

    samples = segment_sampler.generate(policy_model, num_segments)

    assert len(samples) == num_segments


def test_no_rollout_necessary_if_buffer_sufficiently_filled(segment_sampler):
    buffer = MagicMock(spec_set=Buffer, **{"__len__.return_value": 200})
    necessary_steps = segment_sampler._calculate_necessary_rollout_steps(num_items=10, buffer_length=len(buffer))
    assert necessary_steps == 0


def test_calculates_correct_number_of_necessary_rollout_steps(segment_sampler):
    buffer = MagicMock(spec_set=Buffer, **{"__len__.return_value": 0})
    necessary_steps = segment_sampler._calculate_necessary_rollout_steps(num_items=10, buffer_length=len(buffer))
    assert necessary_steps == 30


def test_not_all_segment_samples_are_sampled_in_one_iteration(segment_sampler):
    segments = segment_sampler._compute_num_segments_for_iteration(outstanding_segment_samples=50,
                                                                   segments_per_rollout=20)
    assert segments == 20


def test_at_most_desired_number_of_segments_is_sampled(segment_sampler):
    segments = segment_sampler._compute_num_segments_for_iteration(outstanding_segment_samples=10,
                                                                   segments_per_rollout=20)
    assert segments == 10


def test_does_no_more_rollout_steps_than_fit_into_buffer(segment_sampler):
    segments = segment_sampler._compute_num_rollout_steps_for_iteration(outstanding_rollout_steps=50, buffer_size=10)
    assert segments == 10


def test_at_most_desired_num_of_rollout_steps(segment_sampler):
    segments = segment_sampler._compute_num_rollout_steps_for_iteration(outstanding_rollout_steps=20, buffer_size=50)
    assert segments == 20


def test_samples_are_segments(policy_model):
    segment_sampler = SegmentSampler(segment_length=2)

    samples = segment_sampler.generate(policy_model, num_alternatives=1)

    assert isinstance(samples[0], Segment)


def test_samples_have_correct_length(policy_model):
    segment_length = 2
    segment_sampler = SegmentSampler(segment_length)

    samples = segment_sampler.generate(policy_model, num_alternatives=1)

    assert len(samples[0]) == segment_length


def test_segment_sample_is_subsegment_of_buffered_trajectory():
    buffer = Buffer(buffer_size=3)
    for i in range(3):
        obs, act, rew, done, info = i, 1, 1, 1, {}
        buffer.append_step(obs, act, rew, done, info)

    policy_model = Mock()
    policy_model.trajectory_buffer = buffer

    segment_sampler = SegmentSampler(segment_length=2)

    samples = segment_sampler.generate(policy_model, num_alternatives=1)

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