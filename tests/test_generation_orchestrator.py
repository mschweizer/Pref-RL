from unittest.mock import Mock


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
