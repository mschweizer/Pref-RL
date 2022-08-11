from abc import ABC, abstractmethod
from typing import Type

from ..agents.policy import PolicyModel
from pref_rl.agents.pbrl_agent import PbRLAgent
from ..preference_collection.collector import AbstractPreferenceCollector
from ..preference_querying.querent import AbstractPreferenceQuerent
from ..query_generation.generator import AbstractQueryGenerator
from ..query_scheduling.schedule import AbstractQuerySchedule
from ..reward_model_training.trainer import RewardModelTrainer
from ..reward_models.utils import get_model_by_name


class PbRLAgentFactory(ABC):

    def __init__(self, pb_step_freq, reward_training_freq, num_epochs_in_pretraining, num_epochs_in_training,
                 dataset_buffer_size=3000):
        self.pb_step_freq = pb_step_freq
        self.reward_training_freq = reward_training_freq
        self.num_epochs_in_pretraining = num_epochs_in_pretraining
        self.num_epochs_in_training = num_epochs_in_training
        self.dataset_size = dataset_buffer_size

    @staticmethod
    def _create_reward_model(env, reward_model_name):
        """ Returns reward model. """
        reward_model_cls = get_model_by_name(reward_model_name)
        return reward_model_cls(env)

    @abstractmethod
    def _create_policy_model(self, env, reward_model, load_file=None) -> PolicyModel:
        """ Returns policy model. """

    @abstractmethod
    def _create_reward_model_trainer(self, reward_model) -> RewardModelTrainer:
        """ Returns reward model trainer. """

    @abstractmethod
    def _create_pretraining_query_generator(self) -> AbstractQueryGenerator:
        """ Returns query generator for pretraining. """

    @abstractmethod
    def _create_query_generator(self) -> AbstractQueryGenerator:
        """ Returns query generator for main training. """

    @abstractmethod
    def _create_preference_collector(self) -> AbstractPreferenceCollector:
        """ Returns preference collector. """

    @abstractmethod
    def _create_preference_querent(self) -> AbstractPreferenceQuerent:
        """ Returns preference querent. """

    @abstractmethod
    def _create_query_schedule_cls(self) -> Type[AbstractQuerySchedule]:
        """ Returns query schedule class. """

    def create_agent(self, env, reward_model_name, agent_name=None, load_file=None) -> PbRLAgent:
        reward_model = self._create_reward_model(env, reward_model_name)
        policy_model = self._create_policy_model(
            env, reward_model, load_file=load_file)
        pretraining_query_generator = self._create_pretraining_query_generator()
        query_generator = self._create_query_generator()
        preference_collector = self._create_preference_collector()
        preference_querent = self._create_preference_querent()
        reward_model_trainer = self._create_reward_model_trainer(reward_model)
        query_schedule_cls = self._create_query_schedule_cls()

        return PbRLAgent(policy_model, pretraining_query_generator, query_generator, preference_querent,
                         preference_collector, reward_model_trainer, reward_model, query_schedule_cls,
                         self.pb_step_freq, self.reward_training_freq, self.num_epochs_in_pretraining,
                         self.num_epochs_in_training, agent_name)
