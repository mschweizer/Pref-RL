import pytest

from agent.preference_based.sequential.sequential_pbrl_agent import SequentialPbRLAgent


@pytest.fixture()
def sequential_agent(cartpole_env):
    return SequentialPbRLAgent(cartpole_env)


@pytest.mark.parametrize('training', [True, False])
def test_generate_queries(sequential_agent, training):
    num_queries = 3
    num_existing_queries = len(sequential_agent.query_candidates)

    sequential_agent.generate_queries(num_queries, with_policy_training=training)

    assert len(sequential_agent.query_candidates) == num_existing_queries + num_queries


def test_query_preferences(sequential_agent):
    num_preferences = 3
    num_existing_preferences = len(sequential_agent.preferences)
    sequential_agent.generate_queries(num_queries=num_preferences, with_policy_training=False)

    sequential_agent.query_preferences(num_preferences)

    assert len(sequential_agent.preferences) == num_existing_preferences + num_preferences


def test_pb_learn(cartpole_env):
    agent = SequentialPbRLAgent(cartpole_env, num_pretraining_epochs=1, num_training_epochs_per_iteration=1,
                                preferences_per_iteration=1)

    agent.pb_learn(num_training_timesteps=1, num_pretraining_preferences=1)
