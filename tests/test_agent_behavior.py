from unittest.mock import patch, Mock

from agent import LearningAgent
from experience import Experience
from preference_query import PreferenceDataGenerator


def test_agent_learns_policy_for_given_environment(env):
    with patch('policy.PPO'):
        agent = LearningAgent(env, num_stacked_frames=4)
        agent.learn(1000)

        agent.policy.model.learn.assert_called()


# TODO: Expand test to "buffers last N time steps"
def test_agent_buffers_recent_behavior(learning_agent):
    buffer_size = learning_agent.learning_orchestrator.preference_data_generator.segment_sampler.trajectory_buffer.size

    last_observation = learning_agent.policy.environment.reset()
    last_done = False

    experiences = []

    for i in range(buffer_size):
        action = learning_agent.policy.environment.action_space.sample()
        new_observation, reward, new_done, info = learning_agent.policy.environment.step(action)
        experiences.append(Experience(last_observation, action, reward, last_done, info))
        last_observation, last_done = new_observation, new_done

    assert learning_agent.policy.environment.trajectory_buffer.experiences == experiences


def test_agent_samples_trajectory_segment_every_sampling_interval(reward_wrapper):
    interval_length = 10

    learning_agent = LearningAgent(reward_wrapper, sampling_interval=interval_length, segment_length=4,
                                   num_stacked_frames=4, simulation_steps_per_policy_update=interval_length)

    learning_agent.learning_orchestrator.preference_data_generator.generate_sample = \
        Mock(spec_set=PreferenceDataGenerator.generate_sample)

    learning_agent.learn(total_time_steps=interval_length)

    learning_agent.learning_orchestrator.preference_data_generator.generate_sample.assert_called_once()


def test_agent_saves_sampled_trajectory_segment(learning_agent):
    trajectory_segment = [Experience(1), Experience(2)]
    learning_agent.learning_orchestrator.preference_data_generator.segment_sampler.generate_sample = \
        Mock(return_value=trajectory_segment)

    learning_agent.learning_orchestrator.preference_data_generator.generate_sample()

    assert trajectory_segment in learning_agent.learning_orchestrator.preference_data_generator.segment_samples


def test_agent_sets_sufficient_trajectory_buffer_length(reward_wrapper):
    segment_length = 3
    num_stacked_frames = 5
    buffer_size = 10

    learning_agent = LearningAgent(reward_wrapper, segment_length=segment_length, num_stacked_frames=num_stacked_frames,
                                   trajectory_buffer_size=buffer_size)

    assert learning_agent.policy.environment.trajectory_buffer.size >= min(segment_length, num_stacked_frames)


def test_agent_queries_preference_every_query_interval(reward_wrapper, segment_samples):
    interval_length = 10

    learning_agent = LearningAgent(reward_wrapper, sampling_interval=interval_length, query_interval=10,
                                   segment_length=4, num_stacked_frames=4, simulation_steps_per_policy_update=10)

    learning_agent.learning_orchestrator.preference_data_generator.collect_preference = \
        Mock(spec_set=PreferenceDataGenerator.collect_preference)

    learning_agent.learning_orchestrator.preference_data_generator.segment_samples = segment_samples

    learning_agent.learn(total_time_steps=interval_length)

    learning_agent.learning_orchestrator.preference_data_generator.collect_preference.assert_called_once()


def test_agent_saves_queried_preference(reward_wrapper, segment_samples):
    interval_length = 10

    learning_agent = LearningAgent(reward_wrapper, sampling_interval=interval_length, query_interval=10,
                                   segment_length=4, num_stacked_frames=4,
                                   simulation_steps_per_policy_update=interval_length)

    learning_agent.learning_orchestrator.preference_data_generator.segment_samples.extend(segment_samples)

    learning_agent.learn(total_time_steps=interval_length)

    assert len(learning_agent.learning_orchestrator.preference_data_generator.preferences) == 1
