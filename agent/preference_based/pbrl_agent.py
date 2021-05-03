from abc import ABC, abstractmethod
from collections import deque

from agent.rl_agent import RLAgent
from preference_data.dataset import PreferenceDataset
from preference_data.querent.preference_querent import AbstractPreferenceQuerent
from preference_data.query_generation.query_generator import AbstractQueryGenerator
from preference_data.query_selection.query_selector import AbstractQuerySelector
from reward_modeling.models.reward.utils import get_model_by_name
from reward_modeling.reward_trainer import AbstractRewardTrainer
from wrappers.utils import add_internal_env_wrappers


class AbstractPbRLAgent(RLAgent, AbstractQueryGenerator, AbstractQuerySelector, AbstractPreferenceQuerent,
                        AbstractRewardTrainer, ABC):
    def __init__(self, env, reward_model_name="Mlp", dataset_capacity=3000):
        reward_model_class = get_model_by_name(reward_model_name)
        self.reward_model = reward_model_class(env)

        # TODO: make deque len either a function of preferences per iteration or a param
        self.queries = deque(maxlen=700)
        self.preferences = PreferenceDataset(capacity=dataset_capacity)

        RLAgent.__init__(self, env=add_internal_env_wrappers(env=env, reward_model=self.reward_model))

    @abstractmethod
    def learn_reward_model(self, *args, **kwargs):
        pass

    def predict_reward(self, observation):
        return self.reward_model(observation)
