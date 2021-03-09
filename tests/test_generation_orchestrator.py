from collections import deque
from unittest.mock import Mock

import pytest

from data_generation.preference_collector import RewardMaximizingPreferenceCollector
from data_generation.query_generator import RandomQueryGenerator
from data_generation.segment_sampler import TrajectorySegmentSampler
from orchestration.generation_orchestrator import GenerationOrchestrator


@pytest.fixture()
def generation_orchestrator():
    segment_sampler = TrajectorySegmentSampler(deque(maxlen=10), segment_length=5)
    query_generator = RandomQueryGenerator(segment_samples=[])
    preference_collector = RewardMaximizingPreferenceCollector(queries=[])

    return GenerationOrchestrator(segment_sampler=segment_sampler, query_generator=query_generator,
                                  preference_collector=preference_collector)


def test_samples_trajectory_segment_every_sampling_interval(generation_orchestrator, policy_model):
    sample_mock = Mock()

    generation_orchestrator.segment_sampler.save_sample = sample_mock
    generation_orchestrator.sampling_interval = 10

    callbacks = generation_orchestrator.create_callbacks()
    policy_model.learn(total_timesteps=10, callback=callbacks)

    sample_mock.assert_called_once()


def test_generates_query_every_query_interval(generation_orchestrator, policy_model):
    query_mock = Mock()

    generation_orchestrator.query_generator.save_query = query_mock
    generation_orchestrator.query_interval = 10

    callbacks = generation_orchestrator.create_callbacks()
    policy_model.learn(total_timesteps=10, callback=callbacks)

    query_mock.assert_called_once()


def test_collects_preference_every_query_interval(generation_orchestrator, policy_model):
    collection_mock = Mock()

    generation_orchestrator.preference_collector.save_preference = collection_mock
    generation_orchestrator.query_interval = 10

    callbacks = generation_orchestrator.create_callbacks()
    policy_model.learn(total_timesteps=10, callback=callbacks)

    collection_mock.assert_called_once()


def test_is_sampling_step(generation_orchestrator):
    generation_orchestrator.sampling_interval = 2
    assert generation_orchestrator.is_sampling_step(4)
    assert not generation_orchestrator.is_sampling_step(5)


def test_is_query_step(generation_orchestrator):
    generation_orchestrator.query_interval = 2
    assert generation_orchestrator.is_query_step(4)
    assert not generation_orchestrator.is_query_step(5)
