import gym
import pytest

from agent import LearningAgent
from data_generation.experience import Experience, PredictionBuffer, ExperienceBuffer
from data_generation.preference_data_generator import PreferenceDataGenerator
from orchestration.learning_orchestrator import LearningOrchestrator
from policy import Policy
from reward_modeling.reward_predictor import RewardPredictor
from reward_modeling.reward_wrapper import RewardWrapper
from reward_modeling.utils import Preprocessor


@pytest.fixture()
def env():
    return gym.make('CartPole-v1')


@pytest.fixture()
def reward_predictor(env):
    return RewardPredictor(env, trajectory_buffer=PredictionBuffer(size=2, prediction_stack_depth=4),
                           num_stacked_frames=2)


@pytest.fixture()
def reward_wrapper(env, reward_predictor):
    return RewardWrapper(env=env)


@pytest.fixture()
def learning_agent(reward_wrapper):
    return LearningAgent(reward_wrapper, sampling_interval=10, segment_length=4, num_stacked_frames=4,
                         simulation_steps_per_policy_update=5)


@pytest.fixture()
def segment_samples():
    segment_1 = [Experience(observation=1, action=1, done=1, reward=1, info={"original_reward": 1}),
                 Experience(observation=1, action=1, done=1, reward=1, info={"original_reward": 1})]
    segment_2 = [Experience(observation=1, action=1, done=1, reward=1, info={"original_reward": 25}),
                 Experience(observation=1, action=1, done=1, reward=1, info={"original_reward": 25})]

    return [segment_1, segment_2]


@pytest.fixture()
def policy(env):
    return Policy(env=env, simulation_steps_per_update=5)


@pytest.fixture()
def learning_orchestrator(policy):
    return LearningOrchestrator(reward_model=policy.reward_model, trajectory_buffer=policy.trajectory_buffer,
                                sampling_interval=10, query_interval=10, training_interval=10)


@pytest.fixture()
def preference_data_generator():
    return PreferenceDataGenerator(trajectory_buffer=ExperienceBuffer(size=10))


@pytest.fixture()
def preprocessor(env):
    return Preprocessor(env, num_stacked_frames=4)
