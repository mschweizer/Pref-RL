import pytest

from agent_factory.agent_assembler import PbRLAgentAssembler
from agent_factory.rl_teacher_factory import SyntheticRLTeacherFactory


@pytest.fixture()
def agent(cartpole_env):
    return PbRLAgentAssembler.assemble_agent(agent_factory=SyntheticRLTeacherFactory(),
                                             env=cartpole_env,
                                             reward_model_name="Mlp",
                                             num_pretraining_epochs=8,
                                             num_training_iteration_epochs=16)


def test_agent_sets_sufficient_trajectory_buffer_length(agent):
    segment_length = 3
    num_stacked_frames = 5

    assert agent.policy_model.env.trajectory_buffer.size >= min(segment_length, num_stacked_frames)


def test_pb_learn(agent):
    agent.pb_learn(num_training_timesteps=1, num_training_preferences=2, num_pretraining_preferences=1)
