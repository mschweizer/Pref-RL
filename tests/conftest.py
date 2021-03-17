import pytest
from stable_baselines3 import A2C

from agent.agent import Agent
from preference_data.generation.generator import Generator
from preference_data.preference.experience import Experience
from preference_data.preference.label import Label
from reward_modeling.models.choice import Choice
from reward_modeling.models.reward import Reward
from wrappers.internal.reward_predictor import RewardPredictor
from wrappers.utils import create_env


@pytest.fixture()
def cartpole_env():
    return create_env('CartPole-v1', termination_penalty=0)


@pytest.fixture(params=('CartPole-v1', 'Pong-v0'))
def env(request):
    return create_env(env_id=request.param, termination_penalty=0)


@pytest.fixture()
def reward_model(cartpole_env):
    return Reward(cartpole_env)


@pytest.fixture()
def reward_wrapper(cartpole_env, reward_model):
    return RewardPredictor(env=cartpole_env, reward_model=reward_model, trajectory_buffer_size=100)


@pytest.fixture()
def learning_agent(reward_wrapper):
    return Agent(reward_wrapper)


@pytest.fixture()
def segment_samples():
    segment_1 = [Experience(observation=1, action=1, reward=1, done=1, info={"original_reward": 1}),
                 Experience(observation=1, action=1, reward=1, done=1, info={"original_reward": 1})]
    segment_2 = [Experience(observation=1, action=1, reward=1, done=1, info={"original_reward": 25}),
                 Experience(observation=1, action=1, reward=1, done=1, info={"original_reward": 25})]

    return [segment_1, segment_2]


@pytest.fixture()
def preference_data_generator(policy_model):
    return Generator(policy_model=policy_model, segment_length=3)


@pytest.fixture()
def policy_model(cartpole_env):
    return A2C('MlpPolicy', env=cartpole_env, n_steps=10)


@pytest.fixture()
def preference(env):
    # TODO: Return a fixed segment (without running the env!) to make it faster and deterministic
    segment_length = 6
    experiences = []
    env.reset()
    for i in range(segment_length * 2):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        experiences.append(Experience(observation, action, reward, done, info))
    query = [experiences[:segment_length], experiences[segment_length:]]
    return query, Label.LEFT


@pytest.fixture()
def choice_model(reward_model):
    return Choice(reward_model)
