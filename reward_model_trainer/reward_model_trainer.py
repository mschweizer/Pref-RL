from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from reward_model_trainer.choice_model import ChoiceModel
from reward_model_trainer.preference_dataset import PreferenceDataset


class AbstractRewardModelTrainer(ABC):

    @abstractmethod
    def train(self, epochs, pretraining=False, *args, **kwargs):
        pass


class RewardModelTrainer(AbstractRewardModelTrainer):
    def __init__(self, reward_model, batch_size=64, learning_rate=1e-3, summary_writing_interval=16,
                 dataset_capacity=3000):
        AbstractRewardModelTrainer.__init__(self)
        self.choice_model = ChoiceModel(reward_model)
        self.optimizer = optim.Adam(self.choice_model.parameters(), lr=learning_rate)
        self.criterion = F.binary_cross_entropy
        self.batch_size = batch_size
        self.writer = SummaryWriter()
        self.writing_interval = summary_writing_interval
        self.global_training_step = 0
        self.preferences = PreferenceDataset(capacity=dataset_capacity)

    # TODO: Rename param `pretraining` to `reset_training_steps`
    def train(self, epochs, pretraining=False, *args, **kwargs):
        # TODO: Remove preferences parameter and switch dataset=preferences to dataset=self.preferences
        train_loader = torch.utils.data.DataLoader(dataset=self.preferences, batch_size=self.batch_size)

        running_loss = 0.
        for epoch in range(epochs):

            for i, data in enumerate(train_loader, 0):
                queries, choices = data

                self.optimizer.zero_grad()

                choice_predictions = self.choice_model(queries).double()
                loss = self.criterion(choice_predictions, choices)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if self._is_writing_iteration(self.global_training_step):
                    self._write_summary(running_loss, pretraining)
                    running_loss = 0.0

                self.global_training_step += 1

        if pretraining:
            # reset global step after every round of pretraining
            self.global_training_step = 0

    def _is_writing_iteration(self, i):
        return i % self.writing_interval == self.writing_interval - 1

    def _write_summary(self, running_loss, pretraining):
        tag = "reward model loss"
        tag += " (pretraining)" if pretraining else ""
        average_loss = running_loss / self.writing_interval
        self.writer.add_scalar(tag,
                               average_loss,
                               self.global_training_step)
