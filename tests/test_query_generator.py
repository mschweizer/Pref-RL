from unittest.mock import Mock

import pytest

from data_generation.query_generator import RandomQueryGenerator


@pytest.fixture()
def query_generator():
    return RandomQueryGenerator(segment_samples=[])


def test_query_generator_generates_valid_preference_query():
    segment_samples = ["segment1", "segment2", "segment3"]

    query_generator = RandomQueryGenerator(segment_samples=segment_samples)

    query = query_generator.generate_query()

    assert type(query) is list
    assert len(query) is 2
    assert query[0] in segment_samples and query[1] in segment_samples


def test_saves_generated_query(query_generator):
    query = [1, 2]

    query_generator.generate_query = Mock(return_value=query)

    query_generator.save_query()

    assert query in query_generator.queries
