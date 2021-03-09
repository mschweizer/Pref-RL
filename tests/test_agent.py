import gym

from agent import LearningAgent


def test_agent_chooses_action(cartpole_env):
    obs = cartpole_env.reset()
    agent = LearningAgent(cartpole_env)
    action = agent.choose_action(obs)
    assert action is not None


def test_agent_chooses_valid_action(cartpole_env):
    mountaincar_env = gym.make('MountainCar-v0')

    cartpole_agent = LearningAgent(cartpole_env)
    mountaincar_agent = LearningAgent(mountaincar_env)

    cartpole_observation = cartpole_env.reset()
    mountaincar_observation = mountaincar_env.reset()

    cartpole_action = cartpole_agent.choose_action(cartpole_observation)
    mountaincar_action = mountaincar_agent.choose_action(mountaincar_observation)

    assert cartpole_env.action_space.contains(cartpole_action[0])
    assert mountaincar_env.action_space.contains(mountaincar_action[0])


def test_agent_sets_sufficient_trajectory_buffer_length(cartpole_env):
    segment_length = 3
    num_stacked_frames = 5
    buffer_size = 10

    learning_agent = LearningAgent(cartpole_env, segment_length=segment_length, trajectory_buffer_size=buffer_size)

    assert learning_agent.env.trajectory_buffer.maxlen >= min(segment_length, num_stacked_frames)
