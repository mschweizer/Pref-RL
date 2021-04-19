from abc import ABC, abstractmethod

from agent.rl_agent import RLAgent
from preference_data.dataset import Dataset
from preference_data.querent.preference_querent import AbstractPreferenceQuerent
from preference_data.query_generation.query_generator import AbstractQueryGenerator
from preference_data.query_selection.query_selector import AbstractQuerySelector
from reward_modeling.models.reward import Reward
from reward_modeling.reward_trainer import AbstractRewardTrainer
from wrappers.utils import add_internal_env_wrappers


class AbstractPbRLAgent(RLAgent, AbstractQueryGenerator, AbstractQuerySelector, AbstractPreferenceQuerent,
                        AbstractRewardTrainer, ABC):
    def __init__(self, env, dataset_capacity=3000):
        self.reward_model = Reward(env)

        self.queries = []
        self.preferences = Dataset(capacity=dataset_capacity)

        RLAgent.__init__(self, env=add_internal_env_wrappers(env=env, reward_model=self.reward_model))
        AbstractQueryGenerator.__init__(self)
        AbstractQuerySelector.__init__(self)
        AbstractPreferenceQuerent.__init__(self)
        AbstractRewardTrainer.__init__(self)

    @abstractmethod
    def learn_reward_model(self, *args, **kwargs):
        pass

    def predict_reward(self, observation):
        return self.reward_model(observation)
