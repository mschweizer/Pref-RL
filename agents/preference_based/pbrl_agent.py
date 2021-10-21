from abc import ABC, abstractmethod
from collections import deque

from agents.preference_based.dataset import PreferenceDataset
from agents.rl_agent import RLAgent
from models.reward.utils import get_model_by_name
from preference_collection.preference_collector import BaseHumanPreferenceCollectorMixin
from query_generation.segment_queries.segment_query_generator import BaseSegmentQueryGeneratorMixin
from reward_model_training.reward_trainer import RewardTrainerMixin
from wrappers.utils import add_internal_env_wrappers


class AbstractPbRLAgent(RLAgent, BaseSegmentQueryGeneratorMixin, BaseHumanPreferenceCollectorMixin,
                        RewardTrainerMixin,
                        ABC):
    def __init__(self, env, reward_model_name="Mlp", dataset_capacity=4096):
        reward_model_class = get_model_by_name(reward_model_name)
        self.reward_model = reward_model_class(env)

        # TODO: make deque len either a function of preferences per iteration or a param
        self.query_candidates = deque(maxlen=1024)
        self.preferences = PreferenceDataset(capacity=dataset_capacity)

        RLAgent.__init__(self, env=add_internal_env_wrappers(env=env, reward_model=self.reward_model))

        BaseSegmentQueryGeneratorMixin.__init__(self, query_candidates=self.query_candidates,
                                                policy_model=self.policy_model, segment_sampling_interval=50)
        BaseHumanPreferenceCollectorMixin.__init__(self, query_candidates=self.query_candidates,
                                                       preferences=self.preferences, output_path='../')
        RewardTrainerMixin.__init__(self, self.reward_model)

    @abstractmethod
    def pb_learn(self, *args, **kwargs):
        pass

    def predict_reward(self, observation):
        return self.reward_model(observation)
