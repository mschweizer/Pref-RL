import pytest

from preference_data.preference.experience import Experience
from preference_data.preference.label import Label
from preference_data.querent.synchronous.oracle.oracle import RewardMaximizingOracle


@pytest.fixture()
def preference_oracle():
    return RewardMaximizingOracle()


def test_reward_maximizing_oracle_prefers_higher_reward():
    segment_1 = [Experience(observation=1, action=1, reward=1, done=1, info={"original_reward": 0}),
                 Experience(observation=1, action=1, reward=1, done=1, info={"original_reward": 0})]
    segment_2 = [Experience(observation=1, action=1, reward=1, done=1, info={"original_reward": 25}),
                 Experience(observation=1, action=1, reward=1, done=1, info={"original_reward": 25})]
    queries = [[segment_1, segment_2]]

    preference_collector = RewardMaximizingOracle()

    preferences = preference_collector.query_preferences(queries)

    assert preferences[0][1] == Label.RIGHT
