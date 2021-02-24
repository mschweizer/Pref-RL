import gym
import pytest
import torch.utils.data
from stable_baselines3 import PPO

from agent import LearningAgent
from data_generation.experience import Experience, PredictionBuffer, ExperienceBuffer
from data_generation.preference_collector import RewardMaximizingPreferenceCollector
from data_generation.preference_data_generator import PreferenceDataGenerator
from data_generation.preference_label import PreferenceLabel
from data_generation.query_generator import RandomQueryGenerator
from data_generation.segment_sampler import TrajectorySegmentSampler
from orchestration.generation_orchestrator import GenerationOrchestrator
from reward_modeling.preference_dataset import PreferenceDataset
from reward_modeling.reward_model import RewardModel, ChoiceModel
from reward_modeling.reward_predictor import RewardPredictor
from reward_modeling.reward_trainer import RewardTrainer
from reward_modeling.reward_wrapper import RewardWrapper
from reward_modeling.utils import Preprocessor, get_flattened_input_length


@pytest.fixture(params=('CartPole-v1', 'Pong-v0'))
def env(request):
    env_id = request.param
    return gym.make(env_id)


@pytest.fixture()
def reward_predictor(env):
    num_stacked_frames = 2
    reward_model = RewardModel(get_flattened_input_length(num_stacked_frames, env))

    return RewardPredictor(env, trajectory_buffer=PredictionBuffer(size=2, num_stacked_frames=4),
                           num_stacked_frames=num_stacked_frames, reward_model=reward_model)


@pytest.fixture()
def reward_model(env):
    return RewardModel(get_flattened_input_length(num_stacked_frames=4, env=env))


@pytest.fixture()
def reward_wrapper(env, reward_model):
    return RewardWrapper(env=env, reward_model=reward_model, trajectory_buffer_size=100, num_stacked_frames=4)


@pytest.fixture()
def learning_agent(reward_wrapper):
    return LearningAgent(reward_wrapper, segment_length=4, num_stacked_frames=4,
                         simulation_steps_per_policy_update=5)


@pytest.fixture()
def segment_samples():
    segment_1 = [Experience(observation=1, action=1, reward=1, done=1, info={"original_reward": 1}),
                 Experience(observation=1, action=1, reward=1, done=1, info={"original_reward": 1})]
    segment_2 = [Experience(observation=1, action=1, reward=1, done=1, info={"original_reward": 25}),
                 Experience(observation=1, action=1, reward=1, done=1, info={"original_reward": 25})]

    return [segment_1, segment_2]


@pytest.fixture()
def generation_orchestrator(segment_sampler, query_generator, preference_collector):
    return GenerationOrchestrator(segment_sampler=segment_sampler, query_generator=query_generator,
                                  preference_collector=preference_collector)


@pytest.fixture()
def preference_data_generator(policy_model):
    return PreferenceDataGenerator(policy_model=policy_model, segment_length=3)


@pytest.fixture()
def preprocessor(env):
    return Preprocessor(env, num_stacked_frames=4)


@pytest.fixture()
def segment_sampler():
    return TrajectorySegmentSampler(ExperienceBuffer(size=10), segment_length=5)


@pytest.fixture()
def query_generator():
    return RandomQueryGenerator(segment_samples=[])


@pytest.fixture()
def preference_collector():
    return RewardMaximizingPreferenceCollector(queries=[])


@pytest.fixture()
def policy_model(reward_wrapper):
    return PPO('MlpPolicy', env=reward_wrapper, n_steps=10)


@pytest.fixture()
def trajectory_segment(reward_wrapper):
    reward_wrapper.reset()
    for i in range(15):
        reward_wrapper.step(reward_wrapper.action_space.sample())

    return reward_wrapper.trajectory_buffer.experiences[-12:]


@pytest.fixture()
def preference(trajectory_segment):
    query = [trajectory_segment[:6], trajectory_segment[6:]]
    return query, PreferenceLabel.LEFT


@pytest.fixture()
def preferences(preference):
    return [preference, preference, preference]


@pytest.fixture()
def preference_dataset(preferences, env):
    return PreferenceDataset(preferences=preferences, env=env, num_stacked_frames=4)


@pytest.fixture()
def preference_data_loader(preference_dataset):
    return torch.utils.data.DataLoader(dataset=preference_dataset, batch_size=2)


@pytest.fixture()
def choice_model(reward_model):
    return ChoiceModel(reward_model)


@pytest.fixture()
def reward_trainer(reward_model):
    return RewardTrainer(reward_model)
