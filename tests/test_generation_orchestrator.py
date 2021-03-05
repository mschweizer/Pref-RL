from unittest.mock import Mock

import pytest

from data_generation.experience import ExperienceBuffer
from data_generation.preference_collector import RewardMaximizingPreferenceCollector
from data_generation.query_generator import RandomQueryGenerator
from data_generation.segment_sampler import TrajectorySegmentSampler
from orchestration.generation_orchestrator import GenerationOrchestrator


@pytest.fixture()
def generation_orchestrator():
    segment_sampler = TrajectorySegmentSampler(ExperienceBuffer(size=10), segment_length=5)
    query_generator = RandomQueryGenerator(segment_samples=[])
    preference_collector = RewardMaximizingPreferenceCollector(queries=[])

    return GenerationOrchestrator(segment_sampler=segment_sampler, query_generator=query_generator,
                                  preference_collector=preference_collector)


def test_samples_trajectory_segment_every_sampling_interval(generation_orchestrator, policy_model):
    sample_mock = Mock()

    generation_orchestrator.segment_sampler.save_sample = sample_mock

    callbacks = generation_orchestrator.create_callbacks(None, sampling_interval=10)
    policy_model.learn(total_timesteps=10, callback=callbacks)

    sample_mock.assert_called_once()


def test_generates_query_every_query_interval(generation_orchestrator, policy_model):
    sample_mock = Mock()

    generation_orchestrator.query_generator.save_query = sample_mock

    callbacks = generation_orchestrator.create_callbacks(None, query_interval=10)
    policy_model.learn(total_timesteps=10, callback=callbacks)

    sample_mock.assert_called_once()


def test_collects_preference_every_query_interval(generation_orchestrator, policy_model):
    sample_mock = Mock()

    generation_orchestrator.preference_collector.save_preference = sample_mock

    callbacks = generation_orchestrator.create_callbacks(None, query_interval=10)
    policy_model.learn(total_timesteps=10, callback=callbacks)

    sample_mock.assert_called_once()
