import logging
from unittest.mock import MagicMock

import pytest

from .....agents.policy.buffered_model import BufferedPolicyModel
from .....environment_wrappers.internal.trajectory_observation.buffer import Buffer
from .....environment_wrappers.internal.trajectory_observation.observer import TrajectoryObserver
from .....query_generation.choice_set_query.item_generation.segment_item.sampler import AbstractSegmentSampler


class ConcreteSegmentSampler(AbstractSegmentSampler):
    def __init__(self, segment_length):
        super().__init__(segment_length)
        self.logger = logging.getLogger()

    def _sample_segment(self, trajectory_buffer):
        return "segment"


@pytest.fixture()
def segment_sampler():
    return ConcreteSegmentSampler(segment_length=1)


def test_samples_correct_number_of_segments(cartpole_env):
    segment_sampler = ConcreteSegmentSampler(segment_length=1)
    num_segments = 10
    policy_model = BufferedPolicyModel(TrajectoryObserver(cartpole_env, trajectory_buffer_size=1000), train_freq=5)

    samples = segment_sampler.generate(policy_model, num_segments)

    assert len(samples) == num_segments


def test_no_rollout_necessary_if_buffer_sufficiently_filled(segment_sampler):
    buffer = MagicMock(spec_set=Buffer, **{"__len__.return_value": 200})
    necessary_steps = segment_sampler._calculate_necessary_rollout_steps(num_items=10, buffer=buffer)
    assert necessary_steps == 0


def test_calculates_correct_number_of_necessary_rollout_steps(segment_sampler):
    buffer = MagicMock(spec_set=Buffer, **{"__len__.return_value": 0})
    necessary_steps = segment_sampler._calculate_necessary_rollout_steps(num_items=10, buffer=buffer)
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
    segments = segment_sampler._compute_num_rollout_steps_for_iteration(buffer_size=10, outstanding_rollout_steps=50)
    assert segments == 10


def test_at_most_desired_num_of_rollout_steps(segment_sampler):
    segments = segment_sampler._compute_num_rollout_steps_for_iteration(buffer_size=50, outstanding_rollout_steps=20)
    assert segments == 20
