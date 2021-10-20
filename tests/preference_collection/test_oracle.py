from preference_collection.label import Label
from preference_collection.preference_oracle import RewardMaximizingOracleMixin
from wrappers.internal.trajectory_buffer import Segment


def test_reward_maximizing_oracle_prefers_higher_reward():
    segment_1 = Segment(observations=[1, 1], actions=[1, 1], rewards=[1, 1], dones=[1, 1],
                        infos=[{"external_reward": 0}, {"external_reward": 0}])
    segment_2 = Segment(observations=[1, 1], actions=[1, 1], rewards=[1, 1], dones=[1, 1],
                        infos=[{"external_reward": 25}, {"external_reward": 25}])

    assert RewardMaximizingOracleMixin().answer(query=[segment_1, segment_2]) == Label.RIGHT
