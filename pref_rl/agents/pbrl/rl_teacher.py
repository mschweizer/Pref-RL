from .agent import PbRLAgent
from ..policy.buffered_model import ObservedPolicyModel
from ...environment_wrappers.internal.reward_predictor import RewardPredictor
from ...environment_wrappers.internal.reward_standardizer import RewardStandardizer
from ...environment_wrappers.internal.trajectory_observation.observer import FrameTrajectoryObserver, TrajectoryObserver
from ...preference_collection.human.collector import HumanPreferenceCollector
from ...preference_collection.synthetic.collector import SyntheticPreferenceCollector
from ...preference_collection.synthetic.oracle import RewardMaximizingOracle
from ...preference_querying.dummy_querent import DummyPreferenceQuerent
from ...preference_querying.human_querent import HumanPreferenceQuerent
from ...preference_querying.query_selection.selector import RandomQuerySelector
from ...query_generation.choice_set_query.alternative_generation.segment_alternative.no_env_reset_sampler import \
    NoEnvResetSegmentSampler
from ...query_generation.choice_set_query.buffered_generator import BufferedChoiceSetQueryGenerator
from ...query_scheduling.utils import get_schedule_by_name
from ...reward_model_training.trainer import RewardModelTrainer
from ...reward_modeling.utils import get_model_cls_by_name


class SyntheticRLTeacher(PbRLAgent):

    def __init__(self, env, reward_model_type, pb_step_freq, policy_train_freq=5, reward_train_freq=None,
                 query_schedule_type="Annealing", trajectory_buffer_size=2000, query_segment_length=25,
                 query_buffer_size=100, dataset_size=5000, num_epochs_in_pretraining=8, num_epochs_in_training=16):
        reward_model = get_model_cls_by_name(reward_model_type)(env=env)
        policy_model = ObservedPolicyModel(env=self._apply_internal_wrappers(env, trajectory_buffer_size, reward_model),
                                           train_freq=policy_train_freq)

        query_schedule_cls = get_schedule_by_name(query_schedule_type)
        query_generator = BufferedChoiceSetQueryGenerator(
            alternative_generator=NoEnvResetSegmentSampler(segment_length=query_segment_length),
            buffer_size=query_buffer_size)
        preference_querent = DummyPreferenceQuerent(query_selector=RandomQuerySelector())
        preference_collector = SyntheticPreferenceCollector(oracle=RewardMaximizingOracle())
        reward_model_trainer = RewardModelTrainer(reward_model, dataset_buffer_size=dataset_size)

        super().__init__(policy_model, query_generator, preference_querent, preference_collector, reward_model_trainer,
                         reward_model, query_schedule_cls, pb_step_freq, reward_train_freq, num_epochs_in_pretraining,
                         num_epochs_in_training)

    @staticmethod
    def _apply_internal_wrappers(env, buffer_size, reward_model):
        env = FrameTrajectoryObserver(env, trajectory_buffer_size=buffer_size)
        env = RewardPredictor(env, reward_model)
        env = RewardStandardizer(env)
        return env


class RLTeacher(PbRLAgent):
    def __init__(self, env, reward_model_type, pb_step_freq, policy_train_freq=5, reward_train_freq=None,
                 query_schedule_type="Annealing", trajectory_buffer_size=2000, query_segment_length=25,
                 query_buffer_size=100, dataset_size=5000, pref_collect_address="url", video_dir="local", fps=20,
                 num_epochs_in_pretraining=8, num_epochs_in_training=16):
        reward_model = get_model_cls_by_name(reward_model_type)(env=env)
        policy_model = ObservedPolicyModel(env=self._apply_internal_wrappers(env, trajectory_buffer_size, reward_model),
                                           train_freq=policy_train_freq)

        query_schedule_cls = get_schedule_by_name(query_schedule_type)
        query_generator = BufferedChoiceSetQueryGenerator(
            alternative_generator=NoEnvResetSegmentSampler(segment_length=query_segment_length),
            buffer_size=query_buffer_size)
        preference_querent = HumanPreferenceQuerent(query_selector=RandomQuerySelector(),
                                                    pref_collect_address=pref_collect_address,
                                                    video_output_directory=video_dir,
                                                    frames_per_second=fps)
        preference_collector = HumanPreferenceCollector(pref_collect_address=pref_collect_address)
        reward_model_trainer = RewardModelTrainer(reward_model, dataset_buffer_size=dataset_size)

        super().__init__(policy_model, query_generator, preference_querent, preference_collector, reward_model_trainer,
                         reward_model, query_schedule_cls, pb_step_freq, reward_train_freq, num_epochs_in_pretraining,
                         num_epochs_in_training)

    @staticmethod
    def _apply_internal_wrappers(env, buffer_size, reward_model):
        env = TrajectoryObserver(env, trajectory_buffer_size=buffer_size)
        env = RewardPredictor(env, reward_model)
        env = RewardStandardizer(env)
        return env
