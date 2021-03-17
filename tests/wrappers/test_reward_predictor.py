from preference_data.preference.experience import Experience


def test_wrapper_buffers_recent_behavior(reward_wrapper):
    # TODO: Expand test to "buffers last N time steps"

    buffer_size = reward_wrapper.trajectory_buffer.maxlen

    last_observation = reward_wrapper.reset()
    last_done = False

    experiences = []

    for i in range(buffer_size):
        action = reward_wrapper.action_space.sample()
        new_observation, reward, new_done, info = reward_wrapper.step(action)
        experiences.append(Experience(last_observation, action, reward, last_done, info))
        last_observation, last_done = new_observation, new_done

    assert list(reward_wrapper.trajectory_buffer) == experiences
