from abc import ABC, abstractmethod
from collections import deque

from agent.rl_agent import RLAgent
from preference_data.dataset import PreferenceDataset
from preference_data.querent.preference_querent import AbstractPreferenceQuerent
from preference_data.query_generation.query_generator import AbstractQueryGenerator
from reward_modeling.models.reward.utils import get_model_by_name
from reward_modeling.reward_trainer import AbstractRewardTrainer
from wrappers.utils import add_internal_env_wrappers


class AbstractPbRLAgent(RLAgent, AbstractQueryGenerator, AbstractPreferenceQuerent, AbstractRewardTrainer, ABC):
    def __init__(self, env, reward_model_name="Mlp", dataset_capacity=4096):
        reward_model_class = get_model_by_name(reward_model_name)
        self.reward_model = reward_model_class(env)

        # TODO: make deque len either a function of preferences per iteration or a param
        self.query_candidates = deque(maxlen=1024)
        self.preferences = PreferenceDataset(capacity=dataset_capacity)

        RLAgent.__init__(self, env=add_internal_env_wrappers(env=env, reward_model=self.reward_model))
        AbstractPreferenceQuerent.__init__(self, query_candidates=self.query_candidates, preferences=self.preferences)

    @abstractmethod
    def pb_learn(self, *args, **kwargs):
        pass

    def predict_reward(self, observation):
        return self.reward_model(observation)
