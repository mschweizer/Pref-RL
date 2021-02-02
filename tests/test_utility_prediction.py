import torch

from data_generation.experience import Experience, ExperienceBuffer


def test_predict_utility(env, reward_predictor):
    num_stacked_frames = 4
    prediction_buffer = ExperienceBuffer(size=num_stacked_frames)

    env.reset()

    for i in range(num_stacked_frames):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        prediction_buffer.append(Experience(observation, action, reward, done, info))

    predicted_utility = reward_predictor.predict_utility()

    assert predicted_utility.dtype == torch.float32

# def test_training_changes_model_parameters(reward_training_env):
#     """
#     Testing whether parameters change uses code from  / is based on
#     https://github.com/suriyadeepan/torchtest/blob/66a2c8b669aa23601f64e208463e9449ffc135da/torchtest/torchtest.py#L106
#     """
#
#     model = reward_training_env.reward_predictor.utility_model
#     params = [np for np in model.named_parameters() if np[1].requires_grad]
#     initial_params = [(name, p.clone()) for (name, p) in params]
#
#     reward_training_env.train_reward_model()
#
#     for (_, p0), (name, p1) in zip(initial_params, params):
#         assert not torch.equal(p0, p1)
