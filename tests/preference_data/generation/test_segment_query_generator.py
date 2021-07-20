from unittest.mock import patch, Mock

import pytest

from preference_data.query_generation.segment.segment_query_generator import AbstractSegmentQueryGenerator, \
    RandomSegmentQueryGenerator


@pytest.fixture()
@patch.multiple(AbstractSegmentQueryGenerator, __abstractmethods__=set())
def segment_query_generator(policy_model):
    return AbstractSegmentQueryGenerator(query_candidates=[], policy_model=policy_model)


def test_generate_queries(policy_model):
    random_segment_query_generator = RandomSegmentQueryGenerator(query_candidates=[], policy_model=policy_model)
    num_queries = 2
    random_segment_query_generator.generate_queries(num_queries=num_queries)
    assert len(random_segment_query_generator.query_candidates) == num_queries


def test_generation_volume_reached(segment_query_generator):
    samples = ["segment_1", "segment_2", "segment_3"]
    segment_query_generator.segment_samples.extend(samples)
    assert segment_query_generator._generation_volume_is_reached(generation_volume=len(samples))


def test_generation_volume_not_reached(segment_query_generator):
    samples = ["segment_1", "segment_2", "segment_3"]
    segment_query_generator.segment_samples.extend(samples)
    assert not segment_query_generator._generation_volume_is_reached(generation_volume=(len(samples) + 1))


def test_generates_correct_number_of_samples_without_training(segment_query_generator):
    segment_query_generator.segment_length = 3
    segment_query_generator.segment_sampling_interval = 5
    segment_query_generator.draw_segment_sample = Mock(return_value="Segment sample")
    segment_query_generator._generate_segment_samples_without_training(num_samples=2)
    assert len(segment_query_generator.segment_samples) == 2
