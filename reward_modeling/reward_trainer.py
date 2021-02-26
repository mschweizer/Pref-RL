import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from reward_modeling.choice_model import ChoiceModel


class RewardTrainer:
    def __init__(self, reward_model):
        self.choice_model = ChoiceModel(reward_model)
        self.optimizer = optim.Adam(self.choice_model.parameters(), lr=100000)  # TODO: Make learning rate a param
        self.criterion = F.binary_cross_entropy

    def train(self, preference_dataset):
        train_loader = torch.utils.data.DataLoader(dataset=preference_dataset, batch_size=2)

        self.optimizer.zero_grad()

        for i, data in enumerate(train_loader, 0):
            queries, choices = data

            choice_predictions = self.choice_model(queries)
            loss = self.criterion(choice_predictions, choices)

            loss.backward()
            self.optimizer.step()
