from typing import Type

from agent_factory.agent_factory import AbstractAgentFactory
from agents.policy_model import PolicyModel
from agents.preference_based.buffered_policy_model import BufferedPolicyModel
from environment_wrappers.internal.reward_monitor import RewardMonitor
from environment_wrappers.internal.reward_predictor import RewardPredictor
from environment_wrappers.internal.reward_standardizer import RewardStandardizer
from environment_wrappers.internal.trajectory_buffer import TrajectoryBuffer, FrameTrajectoryBuffer
from preference_collector.human_preference.human_preference_collector import HumanPreferenceCollector
from preference_collector.preference_collector import AbstractPreferenceCollector
from preference_collector.synthetic_preference.preference_oracle import RewardMaximizingOracle
from preference_collector.synthetic_preference.synthetic_preference_collector import SyntheticPreferenceCollector
from preference_querent.preference_querent import AbstractPreferenceQuerent, SynchronousPreferenceQuerent, \
    HumanPreferenceQuerent
from preference_querent.query_selector.query_selector import RandomQuerySelector
from query_generator.choice_set.choice_set_generator import ChoiceSetGenerator
from query_generator.choice_set.segment.pretraining_segment_sampler import RandomPretrainingSegmentSampler
from query_generator.choice_set.segment.segment_sampler import RandomSegmentSampler
from query_generator.query_generator import AbstractQueryGenerator
from query_generator.query_item_selector import RandomItemSelector
from query_schedule.query_schedule import AbstractQuerySchedule, ConstantQuerySchedule
from reward_model_trainer.reward_model_trainer import RewardModelTrainer


class RLTeacherFactory(AbstractAgentFactory):
    def __init__(self, segment_length=25):
        super().__init__()
        self.env = None
        self.segment_length = segment_length

    def create_env(self, env):
        env = FrameTrajectoryBuffer(env)
        env = RewardPredictor(env, self.reward_model)
        env = RewardStandardizer(env)
        self.env = RewardMonitor(env)
        return self.env

    def create_policy_model(self) -> PolicyModel:
        return BufferedPolicyModel(self.env)

    def create_reward_model_trainer(self) -> RewardModelTrainer:
        return RewardModelTrainer(self.reward_model)

    def create_pretraining_query_generator(self) -> AbstractQueryGenerator:
        return ChoiceSetGenerator(item_generator=RandomPretrainingSegmentSampler(segment_length=self.segment_length),
                                  item_selector=RandomItemSelector())

    def create_query_generator(self) -> AbstractQueryGenerator:
        return ChoiceSetGenerator(item_generator=RandomSegmentSampler(segment_length=self.segment_length),
                                  item_selector=RandomItemSelector())

    def create_preference_collector(self) -> AbstractPreferenceCollector:
        return HumanPreferenceCollector()

    def create_preference_querent(self) -> AbstractPreferenceQuerent:
        return HumanPreferenceQuerent(query_selector=RandomQuerySelector(), video_root_output_dir="./videofiles/")

    def create_query_schedule_cls(self) -> Type[AbstractQuerySchedule]:
        return ConstantQuerySchedule


class SyntheticRLTeacherFactory(RLTeacherFactory):

    def __init__(self, segment_length=25):
        super().__init__(segment_length=segment_length)
        self.preference_collector = None
        self.reward_model_trainer = None

    def create_env(self, env):
        env = TrajectoryBuffer(env)
        env = RewardPredictor(env, self.reward_model)
        env = RewardStandardizer(env)
        self.env = RewardMonitor(env)
        return self.env

    def create_reward_model_trainer(self) -> RewardModelTrainer:
        self.reward_model_trainer = RewardModelTrainer(self.reward_model)
        return self.reward_model_trainer

    def create_preference_collector(self) -> AbstractPreferenceCollector:
        self.preference_collector = SyntheticPreferenceCollector(oracle=RewardMaximizingOracle())
        return self.preference_collector

    def create_preference_querent(self) -> AbstractPreferenceQuerent:
        # TODO: Change RandomQuerySelector -> MostRecentlyGeneratedSelector (otherwise a lot of duplicates when we
        #  choose e.g. 500 out of 500 at random (with replacement!)
        return SynchronousPreferenceQuerent(query_selector=RandomQuerySelector(),
                                            preference_collector=self.preference_collector,
                                            preferences=self.reward_model_trainer.preferences)
