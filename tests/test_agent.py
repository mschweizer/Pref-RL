from unittest.mock import patch

import gym

from agent import LearningAgent


def test_agent_chooses_action(env):
    obs = env.reset()
    agent = LearningAgent(env)
    action = agent.choose_action(obs)
    assert action is not None


def test_agent_chooses_valid_action(env):
    cartpole_env = env
    mountaincar_env = gym.make('MountainCar-v0')

    cartpole_agent = LearningAgent(cartpole_env)
    mountaincar_agent = LearningAgent(mountaincar_env)

    cartpole_observation = cartpole_env.reset()
    mountaincar_observation = mountaincar_env.reset()

    cartpole_action = cartpole_agent.choose_action(cartpole_observation)
    mountaincar_action = mountaincar_agent.choose_action(mountaincar_observation)

    assert cartpole_env.action_space.contains(cartpole_action[0])
    assert mountaincar_env.action_space.contains(mountaincar_action[0])


# def test_agent(env):
#     experience_buffer = ExperienceBuffer(size=4)
#     reward_predictor = RewardPredictor(experience_buffer, env, training_interval=20)
#
#     sampling_buffer = ExperienceBuffer(size=10)
#     trajectory_sampler = TrajectorySampler(sampling_buffer, sampling_interval=5, trajectory_length=4)
#
#     learning_environment = RewardLearningWrapper(env, sampling_buffer, reward_predictor)
#
#     agent = LearningAgent(learning_environment, reward_predictor, trajectory_sampler)
#
#     agent.learn(1000)
#
#     agent.policy_model.learn.assert_called()


def test_agent_learns_policy_for_given_environment(env):
    with patch('agent.PPO'):
        agent = LearningAgent(env, num_stacked_frames=4)
        agent.learn_policy(1000)

        agent.policy_model.learn.assert_called()


def test_agent_sets_sufficient_trajectory_buffer_length(reward_wrapper):
    segment_length = 3
    num_stacked_frames = 5
    buffer_size = 10

    learning_agent = LearningAgent(reward_wrapper, segment_length=segment_length, num_stacked_frames=num_stacked_frames,
                                   trajectory_buffer_size=buffer_size)

    assert learning_agent.env.trajectory_buffer.size >= min(segment_length, num_stacked_frames)
