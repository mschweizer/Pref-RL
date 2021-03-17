from collections import deque
from unittest.mock import Mock

import pytest

from preference_data.generation.orchestrator import Orchestrator
from preference_data.generation.query_generator import RandomQueryGenerator
from preference_data.generation.segment_sampler import SegmentSampler


@pytest.fixture()
def generation_orchestrator():
    segment_sampler = SegmentSampler(deque(maxlen=10), segment_length=5)
    query_generator = RandomQueryGenerator(segment_samples=[])

    return Orchestrator(segment_sampler=segment_sampler, query_generator=query_generator)


def test_samples_trajectory_segment_every_sampling_interval(generation_orchestrator, policy_model):
    sample_mock = Mock()
    interval = 10

    generation_orchestrator.segment_sampler.save_sample = sample_mock
    generation_orchestrator.sampling_interval = interval

    callbacks = generation_orchestrator.create_callbacks()
    policy_model.learn(total_timesteps=interval, callback=callbacks)

    sample_mock.assert_called_once()


def test_generates_query_every_query_interval(generation_orchestrator, policy_model):
    query_mock = Mock()
    interval = 10

    generation_orchestrator.query_generator.save_query = query_mock
    generation_orchestrator.query_interval = interval

    callbacks = generation_orchestrator.create_callbacks()
    policy_model.learn(total_timesteps=interval, callback=callbacks)

    query_mock.assert_called_once()


def test_is_sampling_step(generation_orchestrator):
    generation_orchestrator.sampling_interval = 2
    assert generation_orchestrator.is_sampling_step(4)
    assert not generation_orchestrator.is_sampling_step(5)


def test_is_query_step(generation_orchestrator):
    generation_orchestrator.query_interval = 2
    assert generation_orchestrator.is_query_step(4)
    assert not generation_orchestrator.is_query_step(5)
