from data_generation.experience import Experience


def test_wrapper_buffers_recent_behavior(learning_agent):
    # TODO: Expand test to "buffers last N time steps"

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
