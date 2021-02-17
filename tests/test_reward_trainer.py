import torch


def test_training_has_effect_on_all_model_parameters(reward_trainer, preference_dataset):
    torch.manual_seed(42)

    params = [param for param in reward_trainer.choice_model.named_parameters() if param[1].requires_grad]
    initial_params = [(name, param.clone()) for (name, param) in params]

    reward_trainer.train(preference_dataset)

    for (_, p0), (name, p1) in zip(initial_params, params):
        assert not torch.equal(p0, p1)
