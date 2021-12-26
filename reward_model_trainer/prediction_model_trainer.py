from abc import ABC, abstractmethod
from collections import deque
from sklearn.utils import resample

from reward_model_trainer.reward_model_trainer import RewardModelTrainer
from reward_model_trainer.preference_dataset import PreferenceDataset


class AbstractPredictionModelTrainer(ABC):

    @abstractmethod
    def train(self, epochs, pretraining=False, *args, **kwargs):
        pass


class StandardPredictionModelTrainer(AbstractPredictionModelTrainer):

    def __init__(self, prediction_model, dataset_capacity=3000):
        self.preferences = PreferenceDataset(capacity=dataset_capacity)
        self.trainer = RewardModelTrainer(prediction_model.atomic_model)

    def train(self, epochs, reset_logging_timesteps_afterwards=False, *args, **kwargs):
        self.trainer.preferences = self.preferences
        self.trainer.train(epochs, reset_logging_timesteps_afterwards, *args, **kwargs)


class EnsemblePredictionModelTrainer(AbstractPredictionModelTrainer):

    def __init__(self, ensemble_predition_model, dataset_capacity=3000):
        self.preferences = PreferenceDataset(capacity=dataset_capacity)
        self.trainers = [RewardModelTrainer(prediction_model) for prediction_model in ensemble_predition_model.models]

    def __getitem__(self, item):
        return self.trainers[item]

    def train(self, epochs, reset_logging_timesteps_afterwards=False, *args, **kwargs):
        for trainer in self.trainers:
            trainer.preferences.queries, trainer.preferences.choices = self.boostrapping(self.preferences)
            trainer.train(epochs, reset_logging_timesteps_afterwards, *args, **kwargs)

    @staticmethod
    def boostrapping(preferences):
        sampled_queries, sampled_choices = resample(preferences.queries, preferences.choices,
                                                    replace=True, n_samples=len(preferences))
        return deque(sampled_queries), deque(sampled_choices)
