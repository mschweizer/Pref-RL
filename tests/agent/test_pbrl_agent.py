from unittest.mock import patch

from agents.preference_based.pbrl_agent import AbstractPbRLAgent


@patch.multiple(AbstractPbRLAgent, __abstractmethods__=set())
def test_agent_sets_sufficient_trajectory_buffer_length(cartpole_env):
    segment_length = 3
    num_stacked_frames = 5

    learning_agent = AbstractPbRLAgent(cartpole_env)

    assert learning_agent.policy_model.get_env().get_attr("trajectory_buffer")[0].size \
           >= min(segment_length, num_stacked_frames)
