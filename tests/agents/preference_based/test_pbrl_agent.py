from agents.preference_based.pbrl_agent import PbRLAgent


def test_agent_sets_sufficient_trajectory_buffer_length(cartpole_env):
    segment_length = 3
    num_stacked_frames = 5

    learning_agent = PbRLAgent(cartpole_env)

    assert learning_agent.policy_model.env.trajectory_buffer.size \
           >= min(segment_length, num_stacked_frames)


def test_pb_learn(cartpole_env):
    agent = PbRLAgent(cartpole_env, num_pretraining_epochs=1, num_training_epochs_per_iteration=1)

    agent.pb_learn(num_training_timesteps=1, num_pretraining_preferences=1)