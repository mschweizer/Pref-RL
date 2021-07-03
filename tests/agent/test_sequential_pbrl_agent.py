import pytest

from agent.preference_based.sequential.sequential_pbrl_agent import SequentialPbRLAgent


@pytest.mark.parametrize('training', [True, False])
def test_collect_preferences(cartpole_env, training):
    agent = SequentialPbRLAgent(cartpole_env)

    num_preferences = 3
    num_existing_preferences = len(agent.preferences)

    agent._collect_preferences(num_preferences=num_preferences, with_policy_training=training)

    assert len(agent.preferences) == num_existing_preferences + num_preferences


def test_learn_reward_model(cartpole_env):
    agent = SequentialPbRLAgent(cartpole_env,
                                num_pretraining_epochs=1,
                                num_training_epochs_per_iteration=1,
                                preferences_per_iteration=1)

    agent.pb_learn(num_training_timesteps=1, num_pretraining_preferences=1)
