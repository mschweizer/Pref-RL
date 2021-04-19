import pytest
from stable_baselines3 import A2C

from agent.rl_agent import RLAgent
from preference_data.preference.experience import Experience
from preference_data.preference.label import Label
from reward_modeling.models.choice import ChoiceModel
from reward_modeling.models.reward import RewardModel
from wrappers.internal.reward_predictor import RewardPredictor
from wrappers.utils import create_env, add_internal_env_wrappers


@pytest.fixture()
def cartpole_env():
    return create_env('CartPole-v1', termination_penalty=0)


@pytest.fixture(params=('CartPole-v1', 'Pong-v0'))
def env(request):
    return create_env(env_id=request.param, termination_penalty=0)


@pytest.fixture()
def reward_wrapper(cartpole_env):
    return RewardPredictor(env=cartpole_env, reward_model=RewardModel(cartpole_env), trajectory_buffer_size=100)


@pytest.fixture()
def learning_agent(reward_wrapper):
    return RLAgent(reward_wrapper)


@pytest.fixture()
def segment_samples():
    segment_1 = [Experience(observation=1, action=1, reward=1, done=1, info={"original_reward": 1}),
                 Experience(observation=1, action=1, reward=1, done=1, info={"original_reward": 1})]
    segment_2 = [Experience(observation=1, action=1, reward=1, done=1, info={"original_reward": 25}),
                 Experience(observation=1, action=1, reward=1, done=1, info={"original_reward": 25})]

    return [segment_1, segment_2]


@pytest.fixture()
def policy_model(cartpole_env):
    return A2C('MlpPolicy', env=add_internal_env_wrappers(cartpole_env, RewardModel(cartpole_env)), n_steps=10)


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
    return ChoiceModel(reward_model)
