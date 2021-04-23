from preference_data.query_generation.segment.segment_query_generator import RandomSegmentQueryGenerator


def test_calculate_num_segment_samples(policy_model):
    segment_generator = RandomSegmentQueryGenerator(policy_model=policy_model)
    num_samples = segment_generator.calculate_num_segment_samples(num_queries=500)
    assert num_samples == 157


def test_calculate_num_segment_samples_for_one_query(policy_model):
    segment_generator = RandomSegmentQueryGenerator(policy_model=policy_model)
    num_samples = segment_generator.calculate_num_segment_samples(num_queries=1)
    assert num_samples == segment_generator.segments_per_query


def test_calculate_num_segment_samples_for_no_queries(policy_model):
    segment_generator = RandomSegmentQueryGenerator(policy_model=policy_model)
    num_samples = segment_generator.calculate_num_segment_samples(num_queries=0)
    assert num_samples == segment_generator.segments_per_query
