import gym
import pytest

from agent import LearningAgent
from experience import ExperienceBuffer
from reward_predictor import RewardPredictor
from wrapper import RewardWrapper


@pytest.fixture()
def env():
    return gym.make('CartPole-v1')


@pytest.fixture()
def reward_predictor(env):
    return RewardPredictor(env, ExperienceBuffer(size=2), frame_stack_depth=2)


@pytest.fixture()
def reward_wrapper(env, reward_predictor):
    return RewardWrapper(env=env, reward_predictor=reward_predictor, trajectory_buffer=ExperienceBuffer(size=10))


@pytest.fixture()
def learning_agent(reward_wrapper):
    return LearningAgent(reward_wrapper, sampling_interval=10, segment_length=4, frame_stack_depth=4,
                         simulation_steps_per_update=10)
