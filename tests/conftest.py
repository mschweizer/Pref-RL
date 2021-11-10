import pytest
from stable_baselines3 import A2C

from agents.rl_agent import RLAgent
from environment_wrappers.internal.reward_predictor import RewardPredictor
from environment_wrappers.internal.trajectory_buffer import Buffer
from environment_wrappers.utils import create_env, add_internal_env_wrappers
from preference_collector.binary_choice import BinaryChoice
from preference_collector.preference import BinaryChoiceSetPreference
from query_generator.query import BinaryChoiceQuery
from reward_model_trainer.choice_model import ChoiceModel
from reward_models.mlp import MlpRewardModel


@pytest.fixture()
def cartpole_env():
    return create_env('CartPole-v1', termination_penalty=0)


@pytest.fixture(params=('CartPole-v1', 'Pong-v0'))
def env(request):
    return create_env(env_id=request.param, termination_penalty=0)


@pytest.fixture()
def pong_env():
    return create_env('Pong-v0', termination_penalty=0)


# TODO: extend to AtariCnnRewardModel
@pytest.fixture(params=[MlpRewardModel])
def reward_wrapper(cartpole_env, request):
    reward_model_class = request.param
    return RewardPredictor(env=cartpole_env, reward_model=reward_model_class(cartpole_env))


@pytest.fixture()
def learning_agent(reward_wrapper):
    return RLAgent(reward_wrapper)


@pytest.fixture(params=[MlpRewardModel])
def policy_model(cartpole_env, request):
    reward_model_class = request.param
    return A2C('MlpPolicy', env=add_internal_env_wrappers(cartpole_env, reward_model_class(cartpole_env)), n_steps=10)


@pytest.fixture()
def preference(env):
    segment_length = 6
    buffer = Buffer(buffer_size=50)
    env.reset()
    for i in range(segment_length * 2):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        buffer.append_step(observation, action, reward, done, info)
    query = BinaryChoiceQuery(choice_set=[buffer.get_segment(0, segment_length),
                                          buffer.get_segment(segment_length, 2 * segment_length)])
    return BinaryChoiceSetPreference(query, BinaryChoice.LEFT)


@pytest.fixture()
def choice_model(reward_model):
    return ChoiceModel(reward_model)
