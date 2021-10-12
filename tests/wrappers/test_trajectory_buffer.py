from wrappers.internal.experience import Experience
from wrappers.internal.trajectory_buffer import TrajectoryBuffer


def test_wrapper_buffers_recent_behavior(cartpole_env):
    wrapper = TrajectoryBuffer(cartpole_env)

    buffer_size = wrapper.trajectory_buffer.maxlen

    last_observation = wrapper.reset()
    last_done = False

    experiences = []

    for i in range(buffer_size):
        action = wrapper.action_space.sample()
        new_observation, reward, new_done, info = wrapper.step(action)
        experiences.append(Experience(last_observation, action, reward, last_done, info))
        last_observation, last_done = new_observation, new_done

    assert list(wrapper.trajectory_buffer) == experiences
