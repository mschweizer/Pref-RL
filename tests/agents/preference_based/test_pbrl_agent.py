import pytest

from agent_factory.rl_teacher_factory import SyntheticRLTeacherFactory


@pytest.fixture()
def agent(cartpole_env):
    factory = SyntheticRLTeacherFactory(policy_train_freq=5, pb_step_freq=100,
                                        num_epochs_in_pretraining=1, num_epochs_in_training=1)
    return factory.create_agent(env=cartpole_env, reward_model_name="Mlp")


def test_agent_sets_sufficient_trajectory_buffer_length(agent):
    segment_length = 3
    num_stacked_frames = 5

    assert min(segment_length, num_stacked_frames) <= agent.policy_model.trajectory_buffer.size


def test_pb_learn(agent):
    agent.pb_learn(num_training_timesteps=1, num_training_preferences=2, num_pretraining_preferences=1)
