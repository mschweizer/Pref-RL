from typing import Type

from agent_factory.agent_factory import AbstractAgentFactory
from agents.policy_model import PolicyModel
from agents.preference_based.buffered_policy_model import BufferedPolicyModel
from environment_wrappers.internal.reward_monitor import RewardMonitor
from environment_wrappers.internal.reward_predictor import RewardPredictor
from environment_wrappers.internal.reward_standardizer import RewardStandardizer
from environment_wrappers.internal.trajectory_buffer import TrajectoryBuffer
from preference_collector.preference_collector import AbstractPreferenceCollector
from preference_collector.synthetic_preference.preference_oracle import RewardMaximizingOracle
from preference_collector.synthetic_preference.synthetic_preference_collector import SyntheticPreferenceCollector
from preference_querent.preference_querent import AbstractPreferenceQuerent
from preference_querent.query_selector.query_selector import RandomQuerySelector
from preference_querent.dummy_preference_querent import DummyPreferenceQuerent
from query_generator.choice_set.choice_set_generator import ChoiceSetGenerator
from query_generator.choice_set.segment.pretraining_segment_sampler import RandomPretrainingSegmentSampler
from query_generator.choice_set.segment.segment_sampler import RandomSegmentSampler
from query_generator.query_generator import AbstractQueryGenerator
from query_generator.query_item_selector import RandomItemSelector
from query_schedule.query_schedule import AbstractQuerySchedule, ConstantQuerySchedule
from reward_model_trainer.reward_model_trainer import RewardModelTrainer


class SyntheticRLTeacherFactory(AbstractAgentFactory):

    def __init__(self, segment_length=25):
        super().__init__()
        self.segment_length = segment_length

    def create_env(self, env, reward_model):
        env = TrajectoryBuffer(env)
        env = RewardPredictor(env, reward_model)
        env = RewardStandardizer(env)
        env = RewardMonitor(env)
        return env

    def create_policy_model(self, env) -> PolicyModel:
        return BufferedPolicyModel(env)

    def create_reward_model_trainer(self, reward_model) -> RewardModelTrainer:
        return RewardModelTrainer(reward_model)

    def create_pretraining_query_generator(self) -> AbstractQueryGenerator:
        return ChoiceSetGenerator(item_generator=RandomPretrainingSegmentSampler(segment_length=self.segment_length),
                                  item_selector=RandomItemSelector())

    def create_query_generator(self) -> AbstractQueryGenerator:
        return ChoiceSetGenerator(item_generator=RandomSegmentSampler(segment_length=self.segment_length),
                                  item_selector=RandomItemSelector())

    def create_preference_collector(self) -> AbstractPreferenceCollector:
        return SyntheticPreferenceCollector(oracle=RewardMaximizingOracle())

    def create_preference_querent(self) -> AbstractPreferenceQuerent:
        return DummyPreferenceQuerent(query_selector=RandomQuerySelector())

    def create_query_schedule_cls(self) -> Type[AbstractQuerySchedule]:
        return ConstantQuerySchedule
