from preference_data.query_generation.segment.utils import is_sampling_step


def test_is_sampling_step():
    sampling_interval = 2
    assert is_sampling_step(4, sampling_interval)
    assert not is_sampling_step(5, sampling_interval)
