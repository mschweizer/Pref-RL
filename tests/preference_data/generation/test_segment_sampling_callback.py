from unittest.mock import Mock, patch

from query_generation.segment_queries.segment_sampler import AbstractSegmentSampler
from query_generation.segment_queries.segment_sampling_callback import SegmentSamplingCallback


@patch.multiple(AbstractSegmentSampler, __abstractmethods__=set())
def test_samples_trajectory_segment_every_sampling_interval(policy_model):
    sample_mock = Mock()
    interval = 10

    segment_sampler = AbstractSegmentSampler(segment_samples=[],
                                             trajectory_buffer=policy_model.env.envs[0].trajectory_buffer)
    segment_sampler.try_to_sample = sample_mock

    callback = SegmentSamplingCallback(segment_sampler=segment_sampler, sampling_interval=interval,
                                       generation_volume=10)

    policy_model.learn(total_timesteps=interval, callback=callback)

    sample_mock.assert_called_once()
