from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from reward_modeling.models.choice import ChoiceModel


class AbstractRewardTrainer(ABC):

    @abstractmethod
    def train_reward_model(self, preferences, epochs, pretraining=False, *args, **kwargs):
        pass


class RewardTrainer(AbstractRewardTrainer):
    def __init__(self, reward_model, batch_size=64, learning_rate=1e-3):
        AbstractRewardTrainer.__init__(self)
        self.choice_model = ChoiceModel(reward_model)
        self.optimizer = optim.Adam(self.choice_model.parameters(), lr=learning_rate)
        self.criterion = F.binary_cross_entropy
        self.batch_size = batch_size
        self.writer = SummaryWriter()

    def train_reward_model(self, preferences, epochs, pretraining=False, *args, **kwargs):
        train_loader = torch.utils.data.DataLoader(dataset=preferences, batch_size=self.batch_size)

        for epoch in range(epochs):

            for i, data in enumerate(train_loader, 0):
                queries, choices = data

                self.optimizer.zero_grad()

                choice_predictions = self.choice_model(queries).double()
                loss = self.criterion(choice_predictions, choices)

                loss.backward()
                self.optimizer.step()

                self._write_summary(epoch, i, batch_loss=loss.item(), pretraining=pretraining)

    def _calculate_iteration(self, epoch, i):
        return epoch * self.batch_size + i

    def _write_summary(self, epoch, i, batch_loss, pretraining):
        iteration = self._calculate_iteration(epoch, i)
        tag = "training loss"
        tag += " (pretraining)" if pretraining else ""
        self.writer.add_scalar(tag, batch_loss / self.batch_size, iteration)
