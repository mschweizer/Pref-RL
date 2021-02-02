from data_generation.experience import ExperienceBuffer
from data_generation.preference_data_generator import PreferenceDataGenerator


def test_query_generator_generates_valid_preference_query():
    segment_samples = ["segment1", "segment2", "segment3"]

    preference_data_generator = PreferenceDataGenerator(trajectory_buffer=ExperienceBuffer(size=10))

    preference_data_generator.segment_samples = segment_samples

    preference_data_generator.generate_query()
    query = preference_data_generator.queries[0]

    assert type(query) is list
    assert len(query) is 2
    assert query[0] in segment_samples and query[1] in segment_samples
