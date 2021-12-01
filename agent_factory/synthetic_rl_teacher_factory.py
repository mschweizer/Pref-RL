from environment_wrappers.internal.reward_monitor import RewardMonitor
from environment_wrappers.internal.reward_predictor import RewardPredictor
from environment_wrappers.internal.reward_standardizer import \
    RewardStandardizer
from environment_wrappers.internal.trajectory_buffer import TrajectoryBuffer
from preference_collector.preference_collector import \
    AbstractPreferenceCollector
from preference_collector.synthetic_preference.preference_oracle import \
    RewardMaximizingOracle
from preference_collector.synthetic_preference.synthetic_preference_collector import \
    SyntheticPreferenceCollector
from preference_querent.dummy_preference_querent import DummyPreferenceQuerent
from preference_querent.preference_querent import AbstractPreferenceQuerent
from preference_querent.query_selector.query_selector import \
    RandomQuerySelector

from agent_factory.rl_teacher_factory import RLTeacherFactory


class SyntheticRLTeacherFactory(RLTeacherFactory):

    def __init__(self, segment_length=25):
        super().__init__(segment_length=segment_length)

    def create_env(self, env, reward_model):
        env = TrajectoryBuffer(env)
        env = RewardPredictor(env, reward_model)
        env = RewardStandardizer(env)
        env = RewardMonitor(env)
        return env

    def create_preference_collector(self) -> AbstractPreferenceCollector:
        self.preference_collector = SyntheticPreferenceCollector(oracle=RewardMaximizingOracle())
        return self.preference_collector

    def create_preference_querent(self) -> AbstractPreferenceQuerent:
        return DummyPreferenceQuerent(query_selector=RandomQuerySelector())
