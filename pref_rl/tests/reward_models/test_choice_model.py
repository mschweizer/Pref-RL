from unittest.mock import MagicMock

import torch

from pref_rl.reward_model_trainer.choice_model import ChoiceModel


def test_compute_choice_probability_from_rewards():
    choice_model = ChoiceModel(MagicMock())
    total_rewards = torch.tensor([[5.0, 5.0]])
    probabilities = choice_model.compute_choice_probability(total_rewards)
    assert probabilities == torch.tensor([.5])
