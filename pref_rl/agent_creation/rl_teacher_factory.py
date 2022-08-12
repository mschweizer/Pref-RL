from typing import Type

from ..agent_creation.agent_factory import PbRLAgentFactory
from ..agents.policy.buffered_model import BufferedPolicyModel
from ..agents.policy.model import PolicyModel
from ..environment_wrappers.internal.reward_monitor import RewardMonitor
from ..environment_wrappers.internal.reward_predictor import RewardPredictor
from ..environment_wrappers.internal.reward_standardizer import RewardStandardizer
from ..environment_wrappers.internal.trajectory_observation.observer import TrajectoryObserver, \
    FrameTrajectoryObserver
from ..preference_collection.collector import AbstractPreferenceCollector
from ..preference_collection.human.collector import HumanPreferenceCollector
from ..preference_collection.synthetic.collector import SyntheticPreferenceCollector
from ..preference_collection.synthetic.oracle import RewardMaximizingOracle
from ..preference_querying.dummy_querent import DummyPreferenceQuerent
from ..preference_querying.human_querent import HumanPreferenceQuerent
from ..preference_querying.querent import AbstractPreferenceQuerent
from ..preference_querying.query_selection.selector import RandomQuerySelector
from ..query_generation.choice_set_query.alternative_generation.segment_alternative.no_env_reset_sampler import \
    NoEnvResetSegmentSampler
from ..query_generation.choice_set_query.random_generator import RandomChoiceSetQueryGenerator
from ..query_generation.generator import AbstractQueryGenerator
from ..query_scheduling.schedule import AbstractQuerySchedule, AnnealingQuerySchedule
from ..reward_model_training.trainer import RewardModelTrainer


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

    def _create_query_generator(self) -> AbstractQueryGenerator:
        return RandomChoiceSetQueryGenerator(
            alternative_generator=NoEnvResetSegmentSampler(segment_length=self.segment_length))

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
