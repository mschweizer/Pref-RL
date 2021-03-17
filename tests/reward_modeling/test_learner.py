import pytest
import torch
from stable_baselines3 import A2C

from preference_data.dataset import Dataset
from reward_modeling.learner import Learner
from reward_modeling.models.reward import Reward
from wrappers.internal.reward_predictor import RewardPredictor


@pytest.fixture()
def preference_dataset(preference, env):
    preferences = [preference, preference, preference]
    return Dataset(capacity=3000, preferences=preferences)


@pytest.fixture()
def reward_learning_agent_cartpole(cartpole_env):
    reward_model = Reward(cartpole_env)
    reward_wrapper = RewardPredictor(env=cartpole_env, reward_model=reward_model, trajectory_buffer_size=100)
    policy_model = A2C('MlpPolicy', env=reward_wrapper, n_steps=10)
    return Learner(policy_model=policy_model, reward_model=reward_model, segment_length=1)


@pytest.fixture()
def reward_learning_agent(env, preference_dataset):
    reward_model = Reward(env)
    reward_wrapper = RewardPredictor(env=env, reward_model=reward_model, trajectory_buffer_size=100)
    policy_model = A2C('MlpPolicy', env=reward_wrapper, n_steps=10)
    reward_learning_agent.preference_dataset = preference_dataset
    return Learner(policy_model=policy_model, reward_model=reward_model, segment_length=6)


def test_fill_dataset(reward_learning_agent_cartpole):
    generation_volume = 3

    reward_learning_agent_cartpole.fill_dataset(generation_volume=generation_volume, with_training=False)

    assert len(reward_learning_agent_cartpole.preference_collector.preferences) == generation_volume


@pytest.mark.skip(reason="Behavior is currently not deterministic. See issue #26")
def test_training_has_effect_on_any_model_parameters(reward_learning_agent):
    """
    Testing whether parameters change uses code from  / is based on
    https://github.com/suriyadeepan/torchtest/blob/66a2c8b669aa23601f64e208463e9449ffc135da/torchtest/torchtest.py#L106
    """

    # TODO: Clarify if the following modifications to torch have an effect on other tests
    torch.manual_seed(42)
    torch.set_deterministic(d=True)

    params = [param for param in reward_learning_agent.choice_model.named_parameters() if param[1].requires_grad]
    initial_params = [(name, param.clone()) for (name, param) in params]

    reward_learning_agent.learn(500)

    param_change = [not torch.equal(p0, p1) for (_, p0), (name, p1) in zip(initial_params, params)]
    assert any(param_change)
