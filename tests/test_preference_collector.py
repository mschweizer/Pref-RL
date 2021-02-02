from data_generation.experience import Experience
from data_generation.preference_collector import RewardMaximizingPreferenceCollector


def test_reward_maximizing_collector_prefers_higher_reward():
    segment_1 = [Experience(observation=1, action=1, done=1, reward=1, info={"original_reward": 0}),
                 Experience(observation=1, action=1, done=1, reward=1, info={"original_reward": 0})]
    segment_2 = [Experience(observation=1, action=1, done=1, reward=1, info={"original_reward": 25}),
                 Experience(observation=1, action=1, done=1, reward=1, info={"original_reward": 25})]
    query = [segment_1, segment_2]

    preference_collector = RewardMaximizingPreferenceCollector()

    preference = preference_collector.collect_preference(query)

    assert preference[0] == segment_2 and preference[1] == segment_1
