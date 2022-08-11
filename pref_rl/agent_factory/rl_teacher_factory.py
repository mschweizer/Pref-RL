from typing import Type

from ..agent_factory.agent_factory import PbRLAgentFactory
from ..agents.policy_model import PolicyModel
from ..agents.preference_based.buffered_policy_model import BufferedPolicyModel
from ..environment_wrappers.internal.reward_monitor import RewardMonitor
from ..environment_wrappers.internal.reward_predictor import RewardPredictor
from ..environment_wrappers.internal.reward_standardizer import RewardStandardizer
from ..environment_wrappers.internal.trajectory_observer.trajectory_observer import TrajectoryObserver, \
    FrameTrajectoryObserver
from ..preference_collector.human_preference.human_preference_collector import HumanPreferenceCollector
from ..preference_collector.preference_collector import AbstractPreferenceCollector
from ..preference_collector.synthetic_preference.preference_oracle import RewardMaximizingOracle
from ..preference_collector.synthetic_preference.synthetic_preference_collector import SyntheticPreferenceCollector
from ..preference_querent.dummy_preference_querent import DummyPreferenceQuerent
from ..preference_querent.human_preference_querent import HumanPreferenceQuerent
from ..preference_querent.preference_querent import AbstractPreferenceQuerent
from ..preference_querent.query_selector.query_selector import RandomQuerySelector
from ..query_generator.choice_set.choice_set_generator import ChoiceSetGenerator
from ..query_generator.choice_set.segment.pretraining_segment_sampler import RandomNoResetPretrainingSegmentSampler
from ..query_generator.choice_set.segment.segment_sampler import RandomNoResetSegmentSampler
from ..query_generator.query_generator import AbstractQueryGenerator
from pref_rl.query_generator.choice_set.query_item_selector import RandomItemSelector
from ..query_schedule.query_schedule import AbstractQuerySchedule, AnnealingQuerySchedule
from ..reward_model_trainer.reward_model_trainer import RewardModelTrainer


class SyntheticRLTeacherFactory(PbRLAgentFactory):

    def __init__(self, policy_train_freq, pb_step_freq, reward_train_freq, num_epochs_in_pretraining,
                 num_epochs_in_training, segment_length=25, dataset_buffer_size=3000):
        super().__init__(pb_step_freq, reward_train_freq, num_epochs_in_pretraining, num_epochs_in_training,
                         dataset_buffer_size)
        self.segment_length = segment_length
        self.policy_train_freq = policy_train_freq

    def _create_policy_model(self, env, reward_model, **kwargs) -> PolicyModel:
        return BufferedPolicyModel(env=self._wrap_env(env, reward_model), train_freq=self.policy_train_freq)

    def _create_reward_model_trainer(self, reward_model) -> RewardModelTrainer:
        return RewardModelTrainer(reward_model, dataset_buffer_size=self.dataset_size)

    def _create_pretraining_query_generator(self) -> AbstractQueryGenerator:
        return ChoiceSetGenerator(item_generator=RandomNoResetPretrainingSegmentSampler(
            segment_length=self.segment_length),
                                  item_selector=RandomItemSelector())

    def _create_query_generator(self) -> AbstractQueryGenerator:
        return ChoiceSetGenerator(item_generator=RandomNoResetSegmentSampler(segment_length=self.segment_length),
                                  item_selector=RandomItemSelector())

    def _create_preference_collector(self) -> AbstractPreferenceCollector:
        return SyntheticPreferenceCollector(oracle=RewardMaximizingOracle())

    def _create_preference_querent(self) -> AbstractPreferenceQuerent:
        return DummyPreferenceQuerent(query_selector=RandomQuerySelector())

    def _create_query_schedule_cls(self) -> Type[AbstractQuerySchedule]:
        return AnnealingQuerySchedule

    def _wrap_env(self, env, reward_model):
        env = TrajectoryObserver(env, trajectory_buffer_size=max(self.pb_step_freq, self.policy_train_freq))
        env = RewardPredictor(env, reward_model)
        env = RewardStandardizer(env)
        env = RewardMonitor(env)
        return env


class RLTeacherFactory(SyntheticRLTeacherFactory):

    def __init__(self, policy_train_freq, pb_step_freq, reward_train_freq, num_epochs_in_pretraining,
                 num_epochs_in_training, pref_collect_address, video_directory, video_segment_length=25,
                 frames_per_second=20, dataset_buffer_size=3000):
        super().__init__(policy_train_freq, pb_step_freq, reward_train_freq, num_epochs_in_pretraining,
                         num_epochs_in_training, segment_length=video_segment_length,
                         dataset_buffer_size=dataset_buffer_size)
        self.pref_collect_address = pref_collect_address
        self.video_directory = video_directory
        self.fps = frames_per_second

    def _create_preference_collector(self) -> AbstractPreferenceCollector:
        return HumanPreferenceCollector(pref_collect_address=self.pref_collect_address)

    def _create_preference_querent(self) -> AbstractPreferenceQuerent:
        return HumanPreferenceQuerent(query_selector=RandomQuerySelector(),
                                      pref_collect_address=self.pref_collect_address,
                                      video_output_directory=self.video_directory,
                                      frames_per_second=self.fps)

    def _wrap_env(self, env, reward_model):
        env = FrameTrajectoryObserver(env, trajectory_buffer_size=max(self.pb_step_freq, self.policy_train_freq))
        env = RewardPredictor(env, reward_model)
        env = RewardStandardizer(env)
        env = RewardMonitor(env)
        return env
