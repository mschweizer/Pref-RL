import pytest
import torch

from data_generation.experience import Experience, ExperienceBuffer, PredictionBuffer
from reward_modeling.reward_model import RewardModel
from reward_modeling.reward_predictor import RewardPredictor
from reward_modeling.utils import get_flattened_input_length


@pytest.fixture()
def reward_predictor(cartpole_env):
    num_stacked_frames = 2
    reward_model = RewardModel(get_flattened_input_length(num_stacked_frames, cartpole_env))

    return RewardPredictor(cartpole_env, trajectory_buffer=PredictionBuffer(size=2, num_stacked_frames=4),
                           num_stacked_frames=num_stacked_frames, reward_model=reward_model)


def test_predict_utility(cartpole_env, reward_predictor):
    prediction_buffer = ExperienceBuffer(size=reward_predictor.num_stacked_frames)

    cartpole_env.reset()

    for i in range(reward_predictor.num_stacked_frames):
        action = cartpole_env.action_space.sample()
        observation, reward, done, info = cartpole_env.step(action)
        prediction_buffer.append(Experience(observation, action, reward, done, info))

    predicted_utility = reward_predictor.predict_utility()

    assert predicted_utility.dtype == torch.float32
