import gym

from agents.rl_agent import RLAgent


def test_agent_chooses_action(cartpole_env):
    obs = cartpole_env.reset()
    agent = RLAgent(cartpole_env)
    action = agent.choose_action(obs)
    assert action is not None


def test_agent_chooses_valid_action(cartpole_env):
    mountaincar_env = gym.make('MountainCar-v0')

    cartpole_agent = RLAgent(cartpole_env)
    mountaincar_agent = RLAgent(mountaincar_env)

    cartpole_observation = cartpole_env.reset()
    mountaincar_observation = mountaincar_env.reset()

    cartpole_action = cartpole_agent.choose_action(cartpole_observation)
    mountaincar_action = mountaincar_agent.choose_action(mountaincar_observation)

    assert cartpole_env.action_space.contains(cartpole_action[0])
    assert mountaincar_env.action_space.contains(mountaincar_action[0])
