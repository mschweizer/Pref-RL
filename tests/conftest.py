import gym
import pytest

from agent import LearningAgent
from experience import Experience, PredictionBuffer
from reward_predictor import RewardPredictor
from wrapper import RewardWrapper


@pytest.fixture()
def env():
    return gym.make('CartPole-v1')


@pytest.fixture()
def reward_predictor(env):
    return RewardPredictor(env,
                           trajectory_buffer=PredictionBuffer(size=2, prediction_stack_depth=4),
                           num_stacked_frames=2,
                           training_interval=10)


@pytest.fixture()
def reward_wrapper(env, reward_predictor):
    return RewardWrapper(env=env, reward_predictor=reward_predictor,
                         trajectory_buffer=PredictionBuffer(size=10, prediction_stack_depth=4))


@pytest.fixture()
def learning_agent(reward_wrapper):
    return LearningAgent(reward_wrapper, sampling_interval=10, segment_length=4, num_stacked_frames=4,
                         simulation_steps_per_update=10)


@pytest.fixture()
def segment_samples():
    segment_1 = [Experience(observation=1, action=1, done=1, reward=1, info={"original_reward": 1}),
                 Experience(observation=1, action=1, done=1, reward=1, info={"original_reward": 1})]
    segment_2 = [Experience(observation=1, action=1, done=1, reward=1, info={"original_reward": 25}),
                 Experience(observation=1, action=1, done=1, reward=1, info={"original_reward": 25})]

    return [segment_1, segment_2]
