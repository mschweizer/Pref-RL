from unittest.mock import patch

import torch

from agents.preference_based.dataset import PreferenceDataset
from models.reward.mlp import MlpRewardModel
from reward_model_training.reward_trainer import RewardTrainer


def test_writes_summary(cartpole_env):
    running_loss = 100

    with patch('reward_model_training.reward_trainer.SummaryWriter'):
        reward_trainer = RewardTrainer(MlpRewardModel(cartpole_env))
        reward_trainer._write_summary(running_loss, pretraining=False)

        reward_trainer.writer.add_scalar.assert_called_with('training loss',
                                                            running_loss / reward_trainer.writing_interval,
                                                            reward_trainer.global_training_step)


def test_is_writing_iteration(cartpole_env):
    reward_model_trainer = RewardTrainer(MlpRewardModel(cartpole_env))
    reward_model_trainer.writing_interval = 10

    # Note: we start counting at 0
    no_writing_iteration = 7
    writing_iteration = 19

    assert reward_model_trainer._is_writing_iteration(writing_iteration)
    assert not reward_model_trainer._is_writing_iteration(no_writing_iteration)


def test_training_has_effect_on_any_model_parameters(env, preference):
    """
    Testing whether parameters change uses code from  / is based on
    https://github.com/suriyadeepan/torchtest/blob/66a2c8b669aa23601f64e208463e9449ffc135da/torchtest/torchtest.py#L106
    """

    reward_trainer = RewardTrainer(reward_model=MlpRewardModel(env), batch_size=4)
    preferences = PreferenceDataset(preferences=[preference, preference, preference, preference])

    # TODO: Clarify if the following modifications to torch have an effect on other tests
    torch.manual_seed(42)
    torch.set_deterministic(d=True)

    params = [param for param in reward_trainer.choice_model.named_parameters() if param[1].requires_grad]
    initial_params = [(name, param.clone()) for (name, param) in params]

    reward_trainer.train_reward_model(preferences=preferences, epochs=1)

    param_change = [not torch.equal(p0, p1) for (_, p0), (name, p1) in zip(initial_params, params)]
    assert any(param_change)
