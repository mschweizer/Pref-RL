from unittest.mock import Mock

from data_generation.experience import Experience


def test_preference_data_generator_saves_sampled_trajectory_segment(preference_data_generator):
    trajectory_segment = [Experience(1), Experience(2)]

    preference_data_generator.segment_sampler.generate_sample = Mock(return_value=trajectory_segment)

    preference_data_generator.generate_sample()

    assert trajectory_segment in preference_data_generator.segment_samples


def test_preference_data_generator_saves_generated_query(preference_data_generator):
    query = [1, 2]

    preference_data_generator.query_generator.generate_query = Mock(return_value=query)

    preference_data_generator.generate_query()

    assert query in preference_data_generator.queries


def test_preference_data_generator_saves_collected_preference(preference_data_generator):
    query = [1, 2]
    preference_order = [2, 1]

    preference_data_generator.query_selector.select_query = Mock(return_value=query)
    preference_data_generator.preference_collector.collect_preference = Mock(return_value=preference_order)

    preference_data_generator.collect_preference()

    assert preference_order in preference_data_generator.preferences
