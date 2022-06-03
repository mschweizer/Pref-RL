from typing import Type

from ..agent_factory.agent_factory import PbRLAgentFactory
from ..agents.policy_model import PolicyModel
from ..agents.preference_based.buffered_policy_model import BufferedPolicyModel
from ..environment_wrappers.internal.reward_monitor import RewardMonitor
from ..environment_wrappers.internal.reward_predictor import RewardPredictor
from ..environment_wrappers.internal.reward_standardizer import RewardStandardizer
from ..environment_wrappers.internal.trajectory_buffer import TrajectoryBuffer, FrameTrajectoryBuffer
from ..preference_collector.human_preference.human_preference_collector import HumanPreferenceCollector
from ..preference_collector.preference_collector import AbstractPreferenceCollector
from ..preference_collector.synthetic_preference.preference_oracle import RewardMaximizingOracle
from ..preference_collector.synthetic_preference.synthetic_preference_collector import SyntheticPreferenceCollector
from ..preference_querent.dummy_preference_querent import DummyPreferenceQuerent
from ..preference_querent.human_preference.human_preference_querent import HumanPreferenceQuerent
from ..preference_querent.preference_querent import AbstractPreferenceQuerent
from ..preference_querent.query_selector.query_selector import RandomQuerySelector
from ..query_generator.choice_set.choice_set_generator import ChoiceSetGenerator
from ..query_generator.choice_set.segment.pretraining_segment_sampler import RandomPretrainingSegmentSampler
from ..query_generator.choice_set.segment.segment_sampler import RandomSegmentSampler
from ..query_generator.query_generator import AbstractQueryGenerator
from ..query_generator.query_item_selector import RandomItemSelector
from ..query_schedule.query_schedule import AbstractQuerySchedule, ConstantQuerySchedule
from ..reward_model_trainer.reward_model_trainer import RewardModelTrainer


class SyntheticRLTeacherFactory(PbRLAgentFactory):

    def __init__(self,
                 policy_train_freq, pb_step_freq, reward_training_freq,
                 num_epochs_in_pretraining, num_epochs_in_training,
                 segment_length=25):
        super().__init__(pb_step_freq, reward_training_freq, num_epochs_in_pretraining, num_epochs_in_training)
        self.segment_length = segment_length
        self.policy_train_freq = policy_train_freq

    def _create_policy_model(self, env, reward_model, **kwargs) -> PolicyModel:
        return BufferedPolicyModel(env=self._wrap_env(env, reward_model), train_freq=self.policy_train_freq)

    def _create_reward_model_trainer(self, reward_model) -> RewardModelTrainer:
        return RewardModelTrainer(reward_model)

    def _create_pretraining_query_generator(self) -> AbstractQueryGenerator:
        return ChoiceSetGenerator(item_generator=RandomPretrainingSegmentSampler(segment_length=self.segment_length),
                                  item_selector=RandomItemSelector())

    def _create_query_generator(self) -> AbstractQueryGenerator:
        return ChoiceSetGenerator(item_generator=RandomSegmentSampler(segment_length=self.segment_length),
                                  item_selector=RandomItemSelector())

    def _create_preference_collector(self) -> AbstractPreferenceCollector:
        return SyntheticPreferenceCollector(oracle=RewardMaximizingOracle())

    def _create_preference_querent(self) -> AbstractPreferenceQuerent:
        return DummyPreferenceQuerent(query_selector=RandomQuerySelector())

    def _create_query_schedule_cls(self) -> Type[AbstractQuerySchedule]:
        return ConstantQuerySchedule

    def _wrap_env(self, env, reward_model):
        env = TrajectoryBuffer(env, trajectory_buffer_size=max(self.pb_step_freq, self.policy_train_freq))
        env = RewardPredictor(env, reward_model)
        env = RewardStandardizer(env)
        env = RewardMonitor(env)
        return env


class RLTeacherFactory(SyntheticRLTeacherFactory):

    def __init__(self, policy_train_freq, pb_step_freq, reward_training_freq, num_epochs_in_pretraining,
                 num_epochs_in_training, segment_length=25, video_output_dir=None):
        super().__init__(policy_train_freq, pb_step_freq, reward_training_freq,
                         num_epochs_in_pretraining, num_epochs_in_training, segment_length=segment_length)
        self.video_output_dir = video_output_dir

    def _create_preference_collector(self) -> AbstractPreferenceCollector:
        return HumanPreferenceCollector()

    def _create_preference_querent(self) -> AbstractPreferenceQuerent:
        if self.video_output_dir is None:
            return HumanPreferenceQuerent(query_selector=RandomQuerySelector())
        else:
            return HumanPreferenceQuerent(query_selector=RandomQuerySelector(),
                                          video_root_output_dir=self.video_output_dir)

    def _wrap_env(self, env, reward_model):
        env = FrameTrajectoryBuffer(env, trajectory_buffer_size=max(self.pb_step_freq, self.policy_train_freq))
        env = RewardPredictor(env, reward_model)
        env = RewardStandardizer(env)
        env = RewardMonitor(env)
        return env
