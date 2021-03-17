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


def test_collects_preferences(preference_collector, segment_samples):
    query = segment_samples
    preference = PreferenceLabel.INDIFFERENT

    num_preferences = len(preference_collector.preferences)

    preference_collector.collect_preference = Mock(return_value=preference)

    preference_collector.collect_preferences(queries=[query])

    assert len(preference_collector.preferences) == num_preferences + 1
