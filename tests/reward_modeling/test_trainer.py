from unittest.mock import patch

import pytest

from reward_modeling.trainer import Trainer


@pytest.fixture()
def reward_model_trainer(reward_model):
    return Trainer(reward_model)


def test_writes_summary(reward_model):
    running_loss = 100
    iteration = 1500

    with patch('reward_modeling.trainer.SummaryWriter'):
        reward_trainer = Trainer(reward_model)
        reward_trainer.write_summary(running_loss, iteration)
        reward_trainer.writer.add_scalar.assert_called_with('training loss',
                                                            running_loss / reward_trainer.writing_interval,
                                                            iteration)


def test_is_writing_iteration(reward_model_trainer):
    reward_model_trainer.writing_interval = 10

    # Note: we start counting at 0
    no_writing_iteration = 7
    writing_iteration = 19

    assert reward_model_trainer._is_writing_iteration(writing_iteration)
    assert not reward_model_trainer._is_writing_iteration(no_writing_iteration)
