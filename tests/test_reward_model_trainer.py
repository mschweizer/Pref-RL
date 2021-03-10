from unittest.mock import patch

import pytest
import torch
from stable_baselines3 import A2C

from reward_modeling.preference_dataset import PreferenceDataset
from reward_modeling.reward_model import RewardModel
from reward_modeling.reward_model_trainer import RewardModelTrainer
from reward_modeling.reward_wrapper import RewardWrapper


@pytest.fixture()
def preference_dataset(preference, env):
    preferences = [preference, preference, preference]
    return PreferenceDataset(capacity=3000, preferences=preferences)


@pytest.fixture()
def reward_trainer_cartpole(cartpole_env):
    reward_model = RewardModel(cartpole_env)
    reward_wrapper = RewardWrapper(env=cartpole_env, reward_model=reward_model, trajectory_buffer_size=100)
    policy_model = A2C('MlpPolicy', env=reward_wrapper, n_steps=10)
    return RewardModelTrainer(policy_model=policy_model, reward_model=reward_model, segment_length=1)


@pytest.fixture()
def reward_trainer(env, preference_dataset):
    reward_model = RewardModel(env)
    reward_wrapper = RewardWrapper(env=env, reward_model=reward_model, trajectory_buffer_size=100)
    policy_model = A2C('MlpPolicy', env=reward_wrapper, n_steps=10)
    reward_trainer.preference_dataset = preference_dataset
    return RewardModelTrainer(policy_model=policy_model, reward_model=reward_model, segment_length=6)


def test_fill_dataset(reward_trainer_cartpole):
    generation_volume = 3

    reward_trainer_cartpole.fill_dataset(generation_volume=generation_volume, with_training=False)

    assert len(reward_trainer_cartpole.preference_dataset) == generation_volume


@pytest.mark.skip(reason="Behavior is currently not deterministic. See issue #26")
def test_training_has_effect_on_any_model_parameters(reward_trainer):
    """
    Testing whether parameters change uses code from  / is based on
    https://github.com/suriyadeepan/torchtest/blob/66a2c8b669aa23601f64e208463e9449ffc135da/torchtest/torchtest.py#L106
    """

    # TODO: Clarify if the following modifications to torch have an effect on other tests
    torch.manual_seed(42)
    torch.set_deterministic(d=True)

    params = [param for param in reward_trainer.choice_model.named_parameters() if param[1].requires_grad]
    initial_params = [(name, param.clone()) for (name, param) in params]

    reward_trainer.train()

    param_change = [not torch.equal(p0, p1) for (_, p0), (name, p1) in zip(initial_params, params)]
    assert any(param_change)


def test_writes_summary(reward_trainer_cartpole, reward_wrapper):
    running_loss = 100
    reward_trainer_cartpole.writing_interval = 10
    iteration = 1500

    with patch('reward_modeling.reward_model_trainer.SummaryWriter'):
        reward_trainer = RewardModelTrainer(policy_model=A2C('MlpPolicy', env=reward_wrapper, n_steps=10),
                                            reward_model=reward_wrapper.reward_model,
                                            segment_length=1)
        reward_trainer.write_summary(running_loss, iteration)
        reward_trainer.writer.add_scalar.assert_called_with('training loss',
                                                            running_loss / reward_trainer.writing_interval,
                                                            iteration)


def test_is_writing_iteration(reward_trainer_cartpole):
    reward_trainer_cartpole.writing_interval = 10

    # Note: we start counting at 0
    no_writing_iteration = 7
    writing_iteration = 19

    assert reward_trainer_cartpole._is_writing_iteration(writing_iteration)
    assert not reward_trainer_cartpole._is_writing_iteration(no_writing_iteration)
