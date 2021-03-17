import gym

from agent.agent import Agent


def test_agent_chooses_action(cartpole_env):
    obs = cartpole_env.reset()
    agent = Agent(cartpole_env)
    action = agent.choose_action(obs)
    assert action is not None


def test_agent_chooses_valid_action(cartpole_env):
    mountaincar_env = gym.make('MountainCar-v0')

    cartpole_agent = Agent(cartpole_env)
    mountaincar_agent = Agent(mountaincar_env)

    cartpole_observation = cartpole_env.reset()
    mountaincar_observation = mountaincar_env.reset()

    cartpole_action = cartpole_agent.choose_action(cartpole_observation)
    mountaincar_action = mountaincar_agent.choose_action(mountaincar_observation)

    assert cartpole_env.action_space.contains(cartpole_action[0])
    assert mountaincar_env.action_space.contains(mountaincar_action[0])


def test_agent_sets_sufficient_trajectory_buffer_length(cartpole_env):
    segment_length = 3
    num_stacked_frames = 5

    learning_agent = Agent(cartpole_env)

    assert learning_agent.policy_model.env.envs[0].trajectory_buffer.maxlen >= min(segment_length, num_stacked_frames)
