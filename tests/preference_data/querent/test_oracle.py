from preference_collection.label import Label
from preference_collection.preference_oracle import RewardMaximizingOracleMixin
from wrappers.internal.experience import Experience


def test_reward_maximizing_oracle_prefers_higher_reward():
    segment_1 = [Experience(observation=1, action=1, reward=1, done=1, info={"external_reward": 0}),
                 Experience(observation=1, action=1, reward=1, done=1, info={"external_reward": 0})]
    segment_2 = [Experience(observation=1, action=1, reward=1, done=1, info={"external_reward": 25}),
                 Experience(observation=1, action=1, reward=1, done=1, info={"external_reward": 25})]

    assert RewardMaximizingOracleMixin().answer(query=[segment_1, segment_2]) == Label.RIGHT
