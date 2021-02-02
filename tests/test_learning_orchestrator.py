from unittest.mock import Mock

from data_generation.preference_data_generator import PreferenceDataGenerator
from orchestration.learning_orchestrator import LearningOrchestrator
from reward_modeling.reward_learner import RewardLearner


def test_learning_orchestrator_samples_trajectory_segment_every_sampling_interval(policy):
    learning_orchestrator = LearningOrchestrator(reward_model=policy.reward_model,
                                                 trajectory_buffer=policy.trajectory_buffer,
                                                 sampling_interval=10)

    sample_mock = Mock(spec_set=PreferenceDataGenerator.generate_sample)

    learning_orchestrator.preference_data_generator.generate_sample = sample_mock

    callbacks = learning_orchestrator.create_callbacks()
    policy.model.learn(total_timesteps=10, callback=callbacks)

    sample_mock.assert_called_once()


def test_learning_orchestrator_queries_preference_every_query_interval(policy):
    learning_orchestrator = LearningOrchestrator(reward_model=policy.reward_model,
                                                 trajectory_buffer=policy.trajectory_buffer,
                                                 query_interval=10)

    sample_mock = Mock(spec_set=PreferenceDataGenerator.collect_preference)

    learning_orchestrator.preference_data_generator.collect_preference = sample_mock

    callbacks = learning_orchestrator.create_callbacks()
    policy.model.learn(total_timesteps=10, callback=callbacks)

    sample_mock.assert_called_once()


def test_learning_orchestrator_trains_reward_model_every_training_interval(policy):
    learning_orchestrator = LearningOrchestrator(reward_model=policy.reward_model,
                                                 trajectory_buffer=policy.trajectory_buffer,
                                                 training_interval=10)

    sample_mock = Mock(spec_set=RewardLearner.learn)

    learning_orchestrator.reward_learner.learn = sample_mock

    callbacks = learning_orchestrator.create_callbacks()
    policy.model.learn(total_timesteps=10, callback=callbacks)

    sample_mock.assert_called_once()
