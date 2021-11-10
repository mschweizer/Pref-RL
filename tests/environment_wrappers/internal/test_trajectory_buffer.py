from environment_wrappers.internal.trajectory_buffer import TrajectoryBuffer


def test_wrapper_buffers_recent_behavior(cartpole_env):
    wrapper = TrajectoryBuffer(cartpole_env)

    buffer_size = wrapper.trajectory_buffer.size

    last_observation = wrapper.reset()

    observations = []

    for i in range(buffer_size):
        action = wrapper.action_space.sample()
        new_observation, _, _, _ = wrapper.step(action)
        observations.append(last_observation)
        last_observation = new_observation

    assert list(wrapper.trajectory_buffer.observations) == observations
