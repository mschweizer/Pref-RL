from abc import ABC, abstractmethod
import copy
import numpy as np




class PredictionModel(ABC):

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass


class StandardPreditionModel(PredictionModel):

    def __init__(self, atomic_model):
        self.atomic_model = atomic_model

    def __len__(self):
        return 1 # hard coded, for purpose of creating prediction_model_trainer

    def predict(self, *args, **kwargs):
        return self.atomic_model(*args, **kwargs)


class EnsemblePredictionModel(PredictionModel):

    def __init__(self, atomic_model, ensemble_size = 3):
        # self.ensemble_size = ensemble_size
        self.models = [copy.copy(atomic_model) for _ in range(ensemble_size)]

    def __getitem__(self, item):
        return self.models[item]

    def __len__(self):
        return len(self.models)

    def predict(self, *args, **kwargs):
        prediction = [model(*args, **kwargs).item() for model in self.models]
        return np.mean(prediction)