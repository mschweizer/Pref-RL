from abc import ABC, abstractmethod
from typing import Type

from agents.policy_model import PolicyModel
from preference_collector.preference_collector import AbstractPreferenceCollector
from preference_querent.preference_querent import AbstractPreferenceQuerent
from query_generator.query_generator import AbstractQueryGenerator
from query_schedule.query_schedule import AbstractQuerySchedule
from reward_model_trainer.reward_model_trainer import RewardModelTrainer
from reward_models.utils import get_model_by_name


class AbstractAgentFactory(ABC):

    def __init__(self):
        self.reward_model = None

    def create_reward_model(self, env, reward_model_name):
        """ Returns reward model. """
        reward_model_cls = get_model_by_name(reward_model_name)
        self.reward_model = reward_model_cls(env)
        return self.reward_model

    @abstractmethod
    def create_env(self, env):
        """ Returns (wrapped) environment. """

    @abstractmethod
    def create_policy_model(self) -> PolicyModel:
        """ Returns policy model. """

    @abstractmethod
    def create_reward_model_trainer(self) -> RewardModelTrainer:
        """ Returns reward model trainer. """

    @abstractmethod
    def create_pretraining_query_generator(self) -> AbstractQueryGenerator:
        """ Returns query generator for pretraining. """

    @abstractmethod
    def create_query_generator(self) -> AbstractQueryGenerator:
        """ Returns query generator for main training. """

    @abstractmethod
    def create_preference_collector(self) -> AbstractPreferenceCollector:
        """ Returns preference collector. """

    @abstractmethod
    def create_preference_querent(self) -> AbstractPreferenceQuerent:
        """ Returns preference querent. """

    @abstractmethod
    def create_query_schedule_cls(self) -> Type[AbstractQuerySchedule]:
        """ Returns query schedule. """
