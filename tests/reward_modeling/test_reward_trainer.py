from unittest.mock import patch

import torch

from preference_data.dataset import PreferenceDataset
from reward_modeling.models.reward import RewardModel
from reward_modeling.reward_trainer import RewardTrainer


def test_writes_summary(cartpole_env):
    batch_loss = 100
    i = 14
    epoch = 3
    batch_size = 64

    with patch('reward_modeling.reward_trainer.SummaryWriter'):
        reward_trainer = RewardTrainer(RewardModel(cartpole_env), batch_size=batch_size)
        reward_trainer._write_summary(epoch=epoch, batch_loss=batch_loss, pretraining=False, i=i)
        reward_trainer.writer.add_scalar.assert_called_with('training loss',
                                                            batch_loss / reward_trainer.batch_size,
                                                            reward_trainer._calculate_iteration(epoch=epoch, i=i))


def test_training_has_effect_on_any_model_parameters(env, preference):
    """
    Testing whether parameters change uses code from  / is based on
    https://github.com/suriyadeepan/torchtest/blob/66a2c8b669aa23601f64e208463e9449ffc135da/torchtest/torchtest.py#L106
    """

    reward_trainer = RewardTrainer(reward_model=RewardModel(env), batch_size=4)
    preferences = PreferenceDataset(preferences=[preference, preference, preference, preference])

    # TODO: Clarify if the following modifications to torch have an effect on other tests
    torch.manual_seed(42)
    torch.set_deterministic(d=True)

    params = [param for param in reward_trainer.choice_model.named_parameters() if param[1].requires_grad]
    initial_params = [(name, param.clone()) for (name, param) in params]

    reward_trainer.train_reward_model(preferences=preferences, epochs=1)

    param_change = [not torch.equal(p0, p1) for (_, p0), (name, p1) in zip(initial_params, params)]
    assert any(param_change)
