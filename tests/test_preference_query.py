from experience import Experience, ExperienceBuffer
from preference_query import PreferenceDataGenerator, RewardMaximizingPreferenceCollector


def test_agent_generates_valid_preference_query():
    segment_samples = ["segment1", "segment2", "segment3"]

    preference_data_generator = PreferenceDataGenerator(trajectory_buffer=ExperienceBuffer(size=10))

    preference_data_generator.segment_samples = segment_samples

    preference_data_generator.generate_query()
    query = preference_data_generator.queries[0]

    assert type(query) is list
    assert len(query) is 2
    assert query[0] in segment_samples and query[1] in segment_samples


def test_higher_reward_is_preferred():
    segment_1 = [Experience(observation=1, action=1, done=1, reward=1, info={"original_reward": 0}),
                 Experience(observation=1, action=1, done=1, reward=1, info={"original_reward": 0})]
    segment_2 = [Experience(observation=1, action=1, done=1, reward=1, info={"original_reward": 25}),
                 Experience(observation=1, action=1, done=1, reward=1, info={"original_reward": 25})]
    queries = [[segment_1, segment_2]]

    preference_collector = RewardMaximizingPreferenceCollector()

    preference = preference_collector.collect_preference(queries)

    assert preference[0] == segment_2 and preference[1] == segment_1
