import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

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

    def train(self, num_epochs=1):
        train_loader = torch.utils.data.DataLoader(dataset=self.preference_dataset, batch_size=2)

        for epoch in range(num_epochs):

            for i, data in enumerate(train_loader, 0):
                queries, choices = data

                self.optimizer.zero_grad()

                choice_predictions = self.choice_model(queries)
                loss = self.criterion(choice_predictions, choices)

                loss.backward()
                self.optimizer.step()

    def fill_dataset(self, generation_volume, with_training=True):
        preferences = self.preference_data_generator.generate(generation_volume=generation_volume,
                                                              with_training=with_training)
        self.preference_dataset.extend(preferences)
