from unittest.mock import Mock, patch

from preference_data.query_generation.segment.segment_query_generator import AbstractSegmentQueryGenerator
from preference_data.query_generation.segment.segment_sampling_callback import SegmentSamplingCallback


@patch.multiple(AbstractSegmentQueryGenerator, __abstractmethods__=set())
def test_samples_trajectory_segment_every_sampling_interval(policy_model):
    sample_mock = Mock()
    interval = 10

    segment_sampler = AbstractSegmentQueryGenerator(policy_model=policy_model)
    segment_sampler.try_to_sample = sample_mock

    callback = SegmentSamplingCallback(segment_sampler=segment_sampler, sampling_interval=interval,
                                       generation_volume=10)

    policy_model.learn(total_timesteps=interval, callback=callback)

    sample_mock.assert_called_once()
