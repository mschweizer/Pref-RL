from unittest.mock import patch, Mock

from agent import LearningAgent
from experience import Experience


def test_agent_learns_policy_for_given_environment(env):
    with patch('agent.PPO'):
        agent = LearningAgent(env, frame_stack_depth=4)
        agent.learn(1000)

        agent.policy_model.learn.assert_called()


# TODO: Expand test to "buffers last N time steps"
def test_agent_buffers_recent_behavior(learning_agent):
    buffer_size = learning_agent.trajectory_sampler.trajectory_buffer.size

    last_observation = learning_agent.environment.reset()
    last_done = False

    experiences = []

    for i in range(buffer_size):
        action = learning_agent.environment.action_space.sample()
        new_observation, reward, new_done, info = learning_agent.environment.step(action)
        experiences.append(Experience(last_observation, action, reward, last_done, info))
        last_observation, last_done = new_observation, new_done

    assert learning_agent.environment.trajectory_buffer.experiences == experiences


def test_agent_samples_trajectory_segment_every_sampling_interval(reward_wrapper):
    interval_length = 10

    learning_agent = LearningAgent(reward_wrapper, sampling_interval=interval_length, segment_length=4,
                                   frame_stack_depth=4,
                                   simulation_steps_per_update=interval_length)

    learning_agent.sample_trajectory = Mock(spec_set=LearningAgent.sample_trajectory)

    learning_agent.learn(total_time_steps=interval_length)

    learning_agent.sample_trajectory.assert_called_once()


def test_agent_saves_sampled_trajectory_segment(learning_agent):
    trajectory_segment = [Experience(1), Experience(2)]
    learning_agent.trajectory_sampler.get_sampled_trajectory = Mock(return_value=trajectory_segment)

    learning_agent.trajectory_sampler.sample_trajectory()

    assert trajectory_segment in learning_agent.trajectory_sampler.samples


def test_agent_sets_sufficient_trajectory_buffer_length(reward_wrapper):
    segment_length = 3
    frame_stack_depth = 5
    buffer_size = 10

    learning_agent = LearningAgent(reward_wrapper, segment_length=segment_length, frame_stack_depth=frame_stack_depth,
                                   trajectory_buffer_size=buffer_size)

    assert learning_agent.trajectory_buffer.size >= min(segment_length, frame_stack_depth)
