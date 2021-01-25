from experience import ExperienceBuffer, Experience


def test_buffer_removes_oldest():
    buffer = ExperienceBuffer(size=2)
    buffer.append(1)
    buffer.append(2)
    buffer.append(3)

    assert buffer.experiences == [2, 3]


def test_experience_equals(env):
    env.reset()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    experience = Experience(observation=observation, action=action, reward=reward, done=done, info=info)
    same_experience = Experience(observation=observation, action=action, reward=reward, done=done, info=info)

    assert experience == same_experience


def test_experience_not_equal(env):
    initial_observation = env.reset()
    experience = Experience(observation=initial_observation)

    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    other_experience = Experience(observation=observation, action=action, reward=reward, done=done, info=info)

    assert experience != other_experience
