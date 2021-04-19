import pytest

from agent.preference_based.sequential.sequential_pbrl_agent import SequentialPbRLAgent


@pytest.mark.parametrize('training', [True, False])
def test_collect_preferences(cartpole_env, training):
    agent = SequentialPbRLAgent(cartpole_env)

    num_preferences = 3
    num_existing_preferences = len(agent.preferences)

    agent.collect_preferences(num_preferences=num_preferences, with_policy_training=training)

    assert len(agent.preferences) == num_existing_preferences + num_preferences
