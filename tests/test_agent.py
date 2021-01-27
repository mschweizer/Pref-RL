from unittest.mock import Mock

import gym

from agent import LearningAgent
from experience import ExperienceBuffer, Experience


def test_choose_action(env):
    obs = env.reset()
    agent = LearningAgent(env)
    action = agent.choose_action(obs)
    assert action is not None


def test_choose_valid_action(env):
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


def test_agent_trains_reward_model_every_training_interval(learning_agent):
    interval_length = 10

    learning_agent.reward_predictor.training_interval = interval_length

    learning_agent.train_reward_model = Mock(spec_set=LearningAgent.train_reward_model)

    learning_agent.learn(total_time_steps=interval_length)

    learning_agent.train_reward_model.assert_called()


def test_samples_subsegment(learning_agent):
    buffer = learning_agent.trajectory_sampler.trajectory_buffer
    learning_agent.trajectory_sampler.segment_length = 2

    buffer.append(1)
    buffer.append(2)
    buffer.append(3)

    segment = learning_agent.trajectory_sampler.get_sampled_trajectory()

    def segment_is_subsegment_of_buffered_experiences(sample_segment):
        first_experience = sample_segment[0]
        most_recent_experience = sample_segment[0]
        for current_experience in sample_segment:
            if current_experience != first_experience and current_experience != most_recent_experience + 1:
                return False
            most_recent_experience = current_experience
        return True

    assert segment_is_subsegment_of_buffered_experiences(segment)


def test_sampled_segment_has_correct_length(learning_agent):
    buffer = ExperienceBuffer(size=3)
    buffer.append(1)
    buffer.append(2)
    buffer.append(3)

    trajectory_sampler = learning_agent.trajectory_sampler

    trajectory_sampler.trajectory_buffer = buffer
    trajectory_sampler.segment_length = 1

    segment_len_1 = trajectory_sampler.get_sampled_trajectory()

    trajectory_sampler.segment_length = 2
    segment_len_2 = trajectory_sampler.get_sampled_trajectory()

    trajectory_sampler.segment_length = 0
    segment_len_0 = trajectory_sampler.get_sampled_trajectory()

    assert len(segment_len_0) == 0
    assert len(segment_len_1) == 1
    assert len(segment_len_2) == 2


def test_trajectories_contain_samples(learning_agent):
    trajectory = [Experience(1), Experience(2), Experience(3), Experience(4)]
    learning_agent.trajectory_sampler.get_sampled_trajectory = Mock(return_value=trajectory)

    learning_agent.learn(total_time_steps=learning_agent.sampling_interval)

    assert learning_agent.segment_samples[0] == trajectory

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
