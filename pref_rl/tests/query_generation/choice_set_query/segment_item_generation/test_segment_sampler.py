import logging
from unittest.mock import Mock, MagicMock

from pref_rl.agents.policy.buffered_model import BufferedPolicyModel
from .....environment_wrappers.internal.trajectory_observation.buffer import Buffer
from .....query_generation.choice_set_query.item_generation.segment_item.sampler import \
    AbstractSegmentSampler, SAMPLING_FAILURE_WARNING_MSG_1


class ConcreteSegmentSampler(AbstractSegmentSampler):
    def __init__(self, segment_length):
        super().__init__(segment_length)
        self.logger = logging.getLogger()

    def _sample_segment(self, trajectory_buffer, segment_length):
        return "sample"


def test_sampler_samples_correct_number_of_samples():
    policy_model = MagicMock(spec_set=BufferedPolicyModel, **{"trajectory_buffer.__len__.return_value": 200})

    segment_sampler = ConcreteSegmentSampler(segment_length=1)
    segment_sampler.logger = MagicMock()
    num_samples = 10

    samples = segment_sampler.generate(policy_model, num_samples)

    assert len(samples) == num_samples


def test_sampler_warns_when_buffer_has_fewer_elements_than_desired_segment_length(caplog):
    policy_model = Mock(spec_set=BufferedPolicyModel)

    # require segment_length of 1 but provide empty buffer
    policy_model.trajectory_buffer = Buffer(buffer_size=1)  # empty buffer
    segment_sampler = ConcreteSegmentSampler(segment_length=1)  # segment_len = 1

    segment_sampler.generate(policy_model, num_items=10)

    assert caplog.records[0].levelname == "WARNING"
    assert SAMPLING_FAILURE_WARNING_MSG_1 in caplog.records[0].message
