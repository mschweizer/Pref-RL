from unittest.mock import Mock

import pytest

from data_generation.experience import Experience
from data_generation.preference_collector import RewardMaximizingPreferenceCollector
from data_generation.preference_label import PreferenceLabel


@pytest.fixture()
def preference_collector():
    return RewardMaximizingPreferenceCollector(queries=[])


def test_reward_maximizing_collector_prefers_higher_reward():
    segment_1 = [Experience(observation=1, action=1, reward=1, done=1, info={"original_reward": 0}),
                 Experience(observation=1, action=1, reward=1, done=1, info={"original_reward": 0})]
    segment_2 = [Experience(observation=1, action=1, reward=1, done=1, info={"original_reward": 25}),
                 Experience(observation=1, action=1, reward=1, done=1, info={"original_reward": 25})]
    query = [segment_1, segment_2]

    preference_collector = RewardMaximizingPreferenceCollector(queries=[])

    preference = preference_collector.collect_preference(query)

    assert preference == PreferenceLabel.RIGHT


def test_saves_collected_preference(preference_collector):
    query = [1, 2]
    preference = PreferenceLabel.INDIFFERENT

    preference_collector.query_selector.select_query = Mock(return_value=query)
    preference_collector.collect_preference = Mock(return_value=preference)

    preference_collector.save_preference()

    assert (query, preference) in preference_collector.preferences
