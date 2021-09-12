import pytest

from query_generation.segment_queries.segment_query_generator import RandomSegmentQueryGeneratorMixin


@pytest.fixture()
def segment_query_generator(policy_model):
    return RandomSegmentQueryGeneratorMixin(policy_model=policy_model, query_candidates=[])


def test_calculate_num_segment_samples(segment_query_generator):
    num_samples = segment_query_generator.calculate_num_segment_samples(num_queries=500)
    assert num_samples == 157


def test_calculate_num_segment_samples_for_one_query(segment_query_generator):
    num_samples = segment_query_generator.calculate_num_segment_samples(num_queries=1)
    assert num_samples == segment_query_generator.segments_per_query


def test_calculate_num_segment_samples_for_no_queries(segment_query_generator):
    num_samples = segment_query_generator.calculate_num_segment_samples(num_queries=0)
    assert num_samples == segment_query_generator.segments_per_query
