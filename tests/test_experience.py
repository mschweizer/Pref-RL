from data_generation.experience import ExperienceBuffer, Experience


def test_buffer_removes_oldest():
    buffer = ExperienceBuffer(size=2)
    buffer.append(1)
    buffer.append(2)
    buffer.append(3)

    assert buffer.experiences == [2, 3]


def test_experience_equals(cartpole_env):
    cartpole_env.reset()
    action = cartpole_env.action_space.sample()
    observation, reward, done, info = cartpole_env.step(action)
    experience = Experience(observation=observation, action=action, reward=reward, done=done, info=info)
    same_experience = Experience(observation=observation, action=action, reward=reward, done=done, info=info)

    assert experience == same_experience


def test_experience_not_equal(cartpole_env):
    initial_observation = cartpole_env.reset()
    experience = Experience(observation=initial_observation)

    action = cartpole_env.action_space.sample()
    observation, reward, done, info = cartpole_env.step(action)
    other_experience = Experience(observation=observation, action=action, reward=reward, done=done, info=info)

    assert experience != other_experience
