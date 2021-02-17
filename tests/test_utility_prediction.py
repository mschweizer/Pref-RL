import torch
import torch.nn.functional as F
import torch.optim as optim

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


def test_training_changes_model_parameters(choice_model, preference_data_loader):
    """
    Testing whether parameters change uses code from  / is based on
    https://github.com/suriyadeepan/torchtest/blob/66a2c8b669aa23601f64e208463e9449ffc135da/torchtest/torchtest.py#L106
    """

    # TODO: Clarify if the following modifications to torch have an effect on other tests
    torch.manual_seed(42)
    torch.set_deterministic(d=True)

    params = [param for param in choice_model.named_parameters() if param[1].requires_grad]
    initial_params = [(name, param.clone()) for (name, param) in params]

    data_iter = iter(preference_data_loader)
    learning_rate = 100000  # large lr to ensure parameters change sufficiently much for the test to succeed
    optimizer = optim.Adam(choice_model.parameters(), lr=learning_rate)
    criterion = F.binary_cross_entropy

    optimizer.zero_grad()
    queries, choices = next(data_iter)

    choice_predictions = choice_model(queries)
    loss = criterion(choice_predictions, choices)

    loss.backward()
    optimizer.step()

    for (_, p0), (name, p1) in zip(initial_params, params):
        assert not torch.equal(p0, p1)
