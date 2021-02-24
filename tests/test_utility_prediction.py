import torch

from data_generation.experience import Experience, ExperienceBuffer


def test_predict_utility(env, reward_predictor):
    prediction_buffer = ExperienceBuffer(size=reward_predictor.num_stacked_frames)

    env.reset()

    for i in range(reward_predictor.num_stacked_frames):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        prediction_buffer.append(Experience(observation, action, reward, done, info))

    predicted_utility = reward_predictor.predict_utility()

    assert predicted_utility.dtype == torch.float32
