import pytest

from agent_factory.agent_assembler import PbRLAgentAssembler
from agent_factory.rl_teacher_factory import SyntheticRLTeacherFactory


@pytest.fixture()
def agent(cartpole_env):
    return PbRLAgentAssembler.assemble_agent(env=cartpole_env, reward_model_name="Mlp",
                                             agent_factory=SyntheticRLTeacherFactory(policy_train_freq=5,
                                                                                     pb_step_freq=100),
                                             num_epochs_in_pretraining=8, num_epochs_in_training=16,
                                             pb_step_freq=100)


def test_agent_sets_sufficient_trajectory_buffer_length(agent):
    segment_length = 3
    num_stacked_frames = 5

    assert agent.policy_model.trajectory_buffer.size >= min(segment_length, num_stacked_frames)


def test_pb_learn(agent):
    agent.pb_learn(num_training_timesteps=1, num_training_preferences=2, num_pretraining_preferences=1)
