import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from data_generation.preference_data_generator import PreferenceDataGenerator
from reward_modeling.choice_model import ChoiceModel
from reward_modeling.preference_dataset import PreferenceDataset


class RewardModelTrainer:
    def __init__(self, policy_model, reward_model, segment_length=25, dataset_capacity=3000):
        self.preference_dataset = PreferenceDataset(capacity=dataset_capacity)
        self.preference_data_generator = PreferenceDataGenerator(policy_model=policy_model,
                                                                 segment_length=segment_length)
        self.choice_model = ChoiceModel(reward_model)
        self.optimizer = optim.Adam(self.choice_model.parameters(), lr=100000)  # TODO: Make learning rate a param
        self.criterion = F.binary_cross_entropy
        self.writer = SummaryWriter()
        self.writing_interval = 10  # TODO: Make writing interval a param

    def train(self, num_epochs=1):
        # TODO: Set sensible batch size value, possibly as param
        train_loader = torch.utils.data.DataLoader(dataset=self.preference_dataset, batch_size=2)

        running_loss = 0.
        for epoch in range(num_epochs):

            for i, data in enumerate(train_loader, 0):
                queries, choices = data

                self.optimizer.zero_grad()

                choice_predictions = self.choice_model(queries).double()
                loss = self.criterion(choice_predictions, choices)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if self._is_writing_iteration(i):
                    iteration = self._calculate_iteration(epoch, i, train_loader)
                    self.write_summary(running_loss, iteration)
                    running_loss = 0.0

    def _is_writing_iteration(self, i):
        return i % self.writing_interval == self.writing_interval - 1

    @staticmethod
    def _calculate_iteration(epoch, i, train_loader):
        return epoch * len(train_loader) + i

    def fill_dataset(self, generation_volume, with_training=True):
        preferences = self.preference_data_generator.generate(generation_volume=generation_volume,
                                                              with_training=with_training)
        self.preference_dataset.extend(preferences)

    def write_summary(self, running_loss, iteration):
        self.writer.add_scalar('training loss',
                               running_loss / self.writing_interval,
                               iteration)
