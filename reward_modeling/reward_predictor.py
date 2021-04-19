from abc import ABC, abstractmethod


class RewardPredictor(ABC):

    @abstractmethod
    def predict_reward(self, observation):
        pass
